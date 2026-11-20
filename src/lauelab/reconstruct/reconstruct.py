# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Wire scan reconstruction functions for Laue analysis."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path
import os

import numpy as np
import subprocess
import re
import shutil
import functools
from importlib import resources

@dataclass
class ReconstructionResult:
    """Result from subprocess or in-process wire-scan reconstruction.

    The first six fields preserve the positional constructor interface of the
    former named tuple. Native-only arrays and stripe timings are ``None`` on
    the subprocess path.

    Attributes
    ----------
    success : bool
        ``True`` when reconstruction completed. Invalid arguments and setup
        failures raise instead of returning a result. Failures after native
        stripe processing starts set this field to ``False`` and preserve
        available partial progress.
    output_files : list of str
        Paths written by this call. For in-process file reconstruction these
        are the per-depth HDF5 files followed by the summary file. The list can
        contain partial output after a runtime failure.
    log : str
        Standard output captured from a subprocess. The in-process path returns
        an empty string.
    error : str or None
        Standard error from an unsuccessful subprocess or the in-process
        runtime error message. ``None`` indicates no reported error.
    command : str
        Executed subprocess command. The in-process path returns ``"liblaue"``.
    return_code : int
        Subprocess exit status. The in-process path returns 0 on success and -1
        for a runtime failure represented by this result.
    images : numpy.ndarray or None
        Retained reconstructed images with shape ``(n_depths, rows, columns)``
        and dtype ``numpy.float64``. These values are not multiplied by the
        integer-output normalization rescale. They are present only when the
        in-process path retains images.
    depth_um : numpy.ndarray or None
        Sample depths with shape ``(n_depths,)`` and dtype ``numpy.float64``, in
        µm along the incident beam relative to the Si origin in the geometry
        file. This field is native-only.
    depth_intensity : numpy.ndarray or None
        Sum of each unscaled reconstructed image, with shape ``(n_depths,)`` and
        dtype ``numpy.float64``. This field is native-only.
    timings : list of StripeTiming or None
        Per-stripe read, native compute, and write timings in seconds. This
        field is native-only.
    last_completed_stripe : int or None
        Zero-based index of the last stripe fully written to every output file,
        or the last stripe computed when no files are written. ``None`` means no
        stripe completed. This field is native-only.
    """

    success: bool
    output_files: List[str]
    log: str
    error: Optional[str] = None
    command: str = ""
    return_code: int = 0
    images: np.ndarray | None = None
    depth_um: np.ndarray | None = None
    depth_intensity: np.ndarray | None = None
    timings: list | None = None
    last_completed_stripe: int | None = None

    def _asdict(self):
        """Return fields as a dictionary, matching the former named tuple."""
        return dict(vars(self))


# Cache the executable path lookup to avoid repeated filesystem searches.
# Finding the executable involves checking package resources and PATH,
# which can be slow. Since the executable location doesn't change during
# program execution, we cache it after the first lookup.
@functools.lru_cache(maxsize=2)
def _find_executable(name: str = 'reconstructN_cpu') -> str:
    """
    Find and cache the reconstruction executable path.
    
    Args:
        name: Name of the executable to find
        
    Returns:
        Path to the executable
        
    Raises:
        FileNotFoundError: If executable cannot be found
    """
    # First try the package bin directory
    try:
        bin_files = resources.files('lauelab.reconstruct.bin')
        exe_path = bin_files / name
        if exe_path.is_file():
            return str(exe_path)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    
    # Try to find in PATH
    system_exe = shutil.which(name)
    if system_exe:
        return system_exe
    
    if name == 'reconstructN_gpu':
        raise FileNotFoundError(
            "Reconstruction executable 'reconstructN_gpu' not found. "
            "GPU reconstruction is not part of the lauelab package build. "
            "Build it separately with CUDA from "
            "src/lauelab/reconstruct/source/recon_gpu (see its README.md) "
            "and put the executable on PATH, or use reconstruct() for the CPU program."
        )
    raise FileNotFoundError(
        f"Reconstruction executable '{name}' not found in the installed lauelab "
        "package or on PATH. Reinstall lauelab so the native build runs, "
        "or put the executable on PATH."
    )


@functools.lru_cache(maxsize=4)
def _validate_executable(exe_path: str) -> None:
    """
    Validate that executable works by running with --help.
    
    Args:
        exe_path: Path to executable
        
    Raises:
        RuntimeError: If executable doesn't work properly
    """
    try:
        result = subprocess.run(
            [exe_path, '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # WireScan should output help text - check for expected output
        # The binary returns exit code 1 for help, so we check stdout instead
        if "Usage: WireScan" not in result.stdout and "Usage: WireScan" not in result.stderr:
            raise RuntimeError(
                f"Executable did not produce expected help output. "
                f"stdout: {result.stdout[:100]}... "
                f"stderr: {result.stderr[:100]}..."
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(
            f"Executable validation failed: {exe_path}. Error: {e}"
        )


def _map_wire_edge(edge: str) -> str:
    """Map user-friendly edge names to program flags."""
    edge_map = {
        'leading': 'l',
        'trailing': 't', 
        'both': 'b',
        'l': 'l',
        't': 't',
        'b': 'b'
    }
    edge_lower = edge.lower()
    if edge_lower not in edge_map:
        raise ValueError(
            f"Invalid wire_edge '{edge}'. "
            "Must be 'leading', 'trailing', or 'both'"
        )
    return edge_map[edge_lower]


def _execute_reconstruction(
    cmd: List[str], 
    output_base: str,
    timeout: int = 7200
) -> ReconstructionResult:
    """
    Execute the reconstruction subprocess.
    
    Args:
        cmd: Command and arguments
        output_base: Base path for output files
        timeout: Timeout in seconds (default 2 hours)
        
    Returns:
        ReconstructionResult with execution details
    """
    try:
        output_dir = Path(output_base).parent
        output_pattern = Path(output_base).name + "*"
        before = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in output_dir.glob(output_pattern)
            if path.is_file()
        } if output_dir.exists() else {}

        # The stand-alone GPU executable links GSL, whose CBLAS library can load
        # OpenBLAS, but it does not call BLAS. Avoid starting OpenBLAS's
        # otherwise spinning worker pool unless the caller opted in. The CPU
        # executable links no BLAS library, so the setting has no effect on it.
        env = os.environ.copy()
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        
        success = result.returncode == 0
        
        output_files = []
        if success and output_dir.exists():
            changed = [
                path
                for path in output_dir.glob(output_pattern)
                if path.is_file()
                and (
                    path not in before
                    or (path.stat().st_mtime_ns, path.stat().st_size) != before[path]
                )
            ]
            output_files = [str(path) for path in sorted(
                changed,
                key=lambda path: [
                    (0, int(part)) if part.isdigit() else (1, part)
                    for part in re.split(r"(\d+)", path.name)
                ],
            )]
        
        return ReconstructionResult(
            success=success,
            output_files=output_files,
            log=result.stdout,
            error=result.stderr if not success else None,
            command=' '.join(cmd),
            return_code=result.returncode
        )
        
    except subprocess.TimeoutExpired:
        return ReconstructionResult(
            success=False,
            output_files=[],
            log='',
            error=f'Process timed out after {timeout} seconds',
            command=' '.join(cmd),
            return_code=-1
        )
    except Exception as e:
        return ReconstructionResult(
            success=False,
            output_files=[],
            log='',
            error=str(e),
            command=' '.join(cmd),
            return_code=-1
        )


def _build_command(
    executable,
    input_file,
    output_file,
    geometry,
    depth_range,
    resolution,
    image_range,
    verbose,
    percent_brightest,
    wire_edge,
    memory_limit_mb,
    normalization,
    output_pixel_type,
    distortion_map,
    detector_number,
):
    if depth_range[0] >= depth_range[1]:
        raise ValueError(
            f"Invalid depth range: start ({depth_range[0]}) must be less than end ({depth_range[1]})"
        )
    command = [
        executable,
        '-i', str(input_file),
        '-o', str(output_file),
        '-g', str(geometry),
        '-s', str(depth_range[0]),
        '-e', str(depth_range[1]),
        '-r', str(resolution),
        '-v', str(verbose),
        '-p', str(percent_brightest),
        '-w', _map_wire_edge(wire_edge),
        '-m', str(memory_limit_mb),
        '-D', str(detector_number),
    ]
    if image_range is not None:
        command.extend(['-f', str(image_range[0]), '-l', str(image_range[1])])
    if normalization:
        command.extend(['-n', normalization])
    if output_pixel_type is not None:
        command.extend(['-t', str(output_pixel_type)])
    if distortion_map:
        command.extend(['-d', distortion_map])
    return command


def reconstruct(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    geometry: Union[str, Path],
    depth_range: Tuple[float, float],
    resolution: float = 1.0,
    *,
    image_range: Optional[Tuple[int, int]] = None,
    verbose: int = 1,
    percent_brightest: float = 100.0,
    wire_edge: str = 'leading',
    memory_limit_mb: int = 8192,
    executable: Optional[str] = None,
    timeout: int = 7200,
    # Advanced options
    normalization: Optional[str] = None,
    output_pixel_type: Optional[int] = None,
    distortion_map: Optional[str] = None,
    detector_number: int = 0,
    wire_depths_file: Optional[str] = None,
    # Threaded version parameters
    num_threads: Optional[int] = None,
    rows_per_stripe: Optional[int] = None,
    cosmic_filter: bool = False,
    norm_exponent: Optional[float] = None,
    norm_threshold: Optional[float] = None
) -> ReconstructionResult:
    """Reconstruct wire-scan data with the native CPU executable.

    Parameters
    ----------
    input_file
        Path to the input HDF5 file.
    output_file
        Base path for output files, without an extension.
    geometry
        Path to the geometry XML file.
    depth_range
        Start and end depths in micrometres. The start must be less than the end.
    resolution
        Depth resolution in micrometres. The default is ``1.0``.
    image_range
        First and last image indices. By default, the executable processes its
        full input range.
    verbose
        Native verbosity level from ``0`` through ``3``. The default is ``1``.
    percent_brightest
        Percentage of the brightest pixels to process. The default is ``100.0``.
    wire_edge
        Wire edge. Use ``"leading"``, ``"trailing"``, or ``"both"``. The
        corresponding short forms ``"l"``, ``"t"``, and ``"b"`` are also accepted.
    memory_limit_mb
        Native memory limit in MB. The default is ``8192``.
    executable
        Path to ``reconstructN_cpu``. By default, the function searches the
        installed package and then ``PATH``.
    timeout
        Process timeout in seconds. The default is ``7200``.
    normalization
        Native normalization tag.
    output_pixel_type
        Native WinView output pixel type from ``0`` through ``7``.
    distortion_map
        Path to a distortion-map file.
    detector_number
        Detector number passed to the native executable. The default is ``0``.
    wire_depths_file
        Path to a file that contains wire-depth corrections.
    num_threads
        Number of OpenMP threads. By default, the executable selects the count.
    rows_per_stripe
        Rows processed per stripe. By default, the executable uses its own value.
    cosmic_filter
        Enable native cosmic-ray filtering.
    norm_exponent
        Exponent for image-intensity scaling.
    norm_threshold
        Threshold for image-intensity scaling.

    Returns
    -------
    ReconstructionResult
        Process status, output paths, captured output, command, and return code.

    Raises
    ------
    FileNotFoundError
        If the CPU executable is not in the installed package or on ``PATH``.
    RuntimeError
        If the executable does not produce the expected help output.
    ValueError
        If ``depth_range`` or ``wire_edge`` is invalid.
    """
    # Find executable
    if executable:
        exe_path = executable
    else:
        exe_path = _find_executable()
    
    # Validate executable on first use
    _validate_executable(exe_path)
    
    cmd = _build_command(
        exe_path, input_file, output_file, geometry, depth_range, resolution,
        image_range, verbose, percent_brightest, wire_edge, memory_limit_mb,
        normalization, output_pixel_type, distortion_map, detector_number,
    )
    if wire_depths_file:
        cmd.extend(['--wireDepths', wire_depths_file])
    
    # Add threaded version parameters
    if num_threads is not None:
        cmd.extend(['-N', str(num_threads)])
    if rows_per_stripe is not None:
        cmd.extend(['-R', str(rows_per_stripe)])
    if cosmic_filter:
        cmd.append('-C')
    if norm_exponent is not None:
        cmd.extend(['-E', str(norm_exponent)])
    if norm_threshold is not None:
        cmd.extend(['-T', str(norm_threshold)])
    
    return _execute_reconstruction(cmd, str(output_file), timeout)




# GPU reconstruction function
def reconstruct_gpu(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    geometry: Union[str, Path],
    depth_range: Tuple[float, float],
    resolution: float = 1.0,
    *,
    image_range: Optional[Tuple[int, int]] = None,
    verbose: int = 1,
    percent_brightest: float = 100.0,
    wire_edge: str = 'leading',
    memory_limit_mb: int = 8192,
    executable: Optional[str] = None,
    timeout: int = 7200,
    # Advanced options
    normalization: Optional[str] = None,
    output_pixel_type: Optional[int] = None,
    distortion_map: Optional[str] = None,
    detector_number: int = 0,
    # GPU-specific parameters
    wire_depths_file: Optional[str] = None,
    cuda_rows: int = 8
) -> ReconstructionResult:
    """Reconstruct wire-scan data with the native CUDA executable.

    The CUDA program does not support cosmic-ray filtering, ``norm_exponent``,
    or ``norm_threshold``. Use :func:`reconstruct` when you need those options.

    Parameters
    ----------
    input_file
        Path to the input HDF5 file.
    output_file
        Base path for output files, without an extension.
    geometry
        Path to the geometry XML file.
    depth_range
        Start and end depths in micrometres. The start must be less than the end.
    resolution
        Depth resolution in micrometres. The default is ``1.0``.
    image_range
        First and last image indices. By default, the executable processes its
        full input range.
    verbose
        Native verbosity level from ``0`` through ``3``. The default is ``1``.
    percent_brightest
        Percentage of the brightest pixels to process. The default is ``100.0``.
    wire_edge
        Wire edge. Use ``"leading"``, ``"trailing"``, or ``"both"``. The
        corresponding short forms ``"l"``, ``"t"``, and ``"b"`` are also accepted.
    memory_limit_mb
        Native memory limit in MB. The default is ``8192``.
    executable
        Path to ``reconstructN_gpu``. By default, the function searches the
        installed package and then ``PATH``.
    timeout
        Process timeout in seconds. The default is ``7200``.
    normalization
        Native normalization tag.
    output_pixel_type
        Native WinView output pixel type from ``0`` through ``7``.
    distortion_map
        Path to a distortion-map file.
    detector_number
        Detector number passed to the native executable. The default is ``0``.
    wire_depths_file
        Path to a file that contains wire-depth corrections for each pixel.
    cuda_rows
        Number of CUDA rows to process. The default is ``8``.

    Returns
    -------
    ReconstructionResult
        Process status, output paths, captured output, command, and return code.

    Raises
    ------
    FileNotFoundError
        If the CUDA executable is not in the installed package or on ``PATH``.
    RuntimeError
        If the executable does not produce the expected help output.
    ValueError
        If ``depth_range`` or ``wire_edge`` is invalid.
    """
    # Find GPU executable
    if executable:
        exe_path = executable
    else:
        exe_path = _find_executable('reconstructN_gpu')
    
    # Validate executable on first use
    _validate_executable(exe_path)
    
    cmd = _build_command(
        exe_path, input_file, output_file, geometry, depth_range, resolution,
        image_range, verbose, percent_brightest, wire_edge, memory_limit_mb,
        normalization, output_pixel_type, distortion_map, detector_number,
    )
    cmd.extend(['-R', str(cuda_rows)])
    if wire_depths_file:
        cmd.extend(['-W', wire_depths_file])  # Note: GPU uses -W, not --wireDepths
    
    return _execute_reconstruction(cmd, str(output_file), timeout)




# Utility functions for common use cases
def find_executable() -> str:
    """Find the native CPU reconstruction executable.

    Returns
    -------
    str
        Path to ``reconstructN_cpu`` in the installed package or on ``PATH``.

    Raises
    ------
    FileNotFoundError
        If the executable cannot be found.
    """
    return _find_executable()


def find_gpu_executable() -> str:
    """Find the native CUDA reconstruction executable.

    Returns
    -------
    str
        Path to ``reconstructN_gpu`` in the installed package or on ``PATH``.

    Raises
    ------
    FileNotFoundError
        If the executable cannot be found.
    """
    return _find_executable('reconstructN_gpu')


def gpu_available() -> bool:
    """Report whether the native CUDA reconstruction executable is usable.

    Returns
    -------
    bool
        ``True`` if ``reconstructN_gpu`` is available and passes validation.
    """
    try:
        exe_path = _find_executable('reconstructN_gpu')
        _validate_executable(exe_path)
        return True
    except (FileNotFoundError, RuntimeError):
        return False
