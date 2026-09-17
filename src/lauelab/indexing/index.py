# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Functional interface for Laue indexing."""

from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from pathlib import Path
import subprocess
import os
import shutil
import functools
import re
from importlib import resources

from lauelab.indexing.lau_dataclasses.step import Step
from lauelab.indexing.lau_dataclasses.indexing import Indexing
from lauelab.indexing.parsers import parse_full_step_data
from lauelab.indexing.xml_utils import write_step_xml, get_default_xml_filename


class IndexingResult(NamedTuple):
    """Result returned by the LaueGo subprocess indexing interface.

    Parameters
    ----------
    success
        Whether the pipeline produced its expected result for the completed
        stages.
    output_files
        Mapping from intermediate output type to generated file path.
    n_peaks_found
        Number of peaks reported by peak search.
    n_indexed
        Number of indexed peaks.
    n_patterns_found
        Number of orientation patterns found.
    indexing_data
        Parsed LaueGo indexing data, if indexing output was available.
    step_data
        Parsed LaueGo XML serialization model, if XML generation succeeded.
    xml_file
        Generated XML path, or `None` when XML was disabled or failed.
    log
        Combined pipeline log.
    error
        Failure description, or `None` when no error was reported.
    command_history
        Subprocess commands and output operations performed by the pipeline.

    Notes
    -----
    This type belongs to the LaueGo subprocess API. New code should use
    :class:`FrameResult` from :func:`index_frame` or :class:`Indexer`.
    """

    success: bool
    output_files: Dict[str, str]
    n_peaks_found: int
    n_indexed: int
    n_patterns_found: int
    indexing_data: Optional[Indexing]
    step_data: Optional[Step]
    xml_file: Optional[str]  # Path to generated XML file
    log: str
    error: Optional[str] = None
    command_history: Tuple[str, ...] = ()


@functools.lru_cache(maxsize=1)
def _find_executables() -> Dict[str, str]:
    """
    Find and cache paths to indexing executables.
    
    Returns:
        Dictionary mapping executable name to its path.
        
    Raises:
        FileNotFoundError: If an executable cannot be found.
    """
    execs = {}
    for name in ['peaksearch', 'pix2qs', 'euler']:
        try:
            bin_files = resources.files('lauelab.indexing.bin')
            exe_path = bin_files / name
            if exe_path.is_file():
                execs[name] = str(exe_path)
                continue
        except (ModuleNotFoundError, FileNotFoundError):
            pass
        
        system_exe = shutil.which(name)
        if system_exe:
            execs[name] = system_exe
            continue
        
        raise FileNotFoundError(
            f"Indexing executable '{name}' not found. "
            "Please compile it first or ensure it's in your PATH."
        )
    return execs


def _run_command(cmd: List[str], timeout: int) -> Tuple[bool, str, str, int, str]:
    """
    Execute a command and capture output.

    Args:
        cmd: Command and arguments.
        timeout: Timeout in seconds.

    Returns:
        A tuple of (success, stdout, stderr, return_code, command_string).
        Note: success is based on return code being 0. Some Laue tools
        have other success codes, which must be handled by the caller.
        This implements graceful degradation - continuing on errors.
    """
    command_string = ' '.join(cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        # Log errors but don't fail immediately - graceful degradation
        return result.returncode == 0, result.stdout, result.stderr, result.returncode, command_string
    except subprocess.TimeoutExpired:
        return False, "", f"Process timed out after {timeout} seconds", -1, command_string
    except FileNotFoundError as e:
        return False, "", f"Executable not found: {e}", -1, command_string
    except Exception as e:
        return False, "", f"An unexpected error occurred: {e}", -1, command_string


def _setup_output_dirs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create the standard output directory structure for indexing.

    Args:
        output_dir: The root output directory.

    Returns:
        A dictionary mapping directory type to its Path object.
    """
    output_dir = Path(output_dir)
    subdirs = {
        'peaks': output_dir / 'peaks',
        'p2q': output_dir / 'p2q',
        'index': output_dir / 'index',
        'xml': output_dir / 'xml',
        'error': output_dir / 'error'
    }
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def _run_peaksearch(input_image: str, output_dir: str,
                   boxsize: int, max_rfactor: float, min_size: int,
                   min_separation: int, threshold: int, peak_shape: str,
                   max_peaks: int, mask_file: Optional[str],
                   threshold_ratio: Optional[float], smooth: bool,
                   timeout: int = 300) -> Tuple[bool, str, str, int, str]:
    """
    Run the peaksearch executable.
    
    Args:
        input_image: Path to input image file.
        output_dir: Directory for output files.
        boxsize: Box size for peak search.
        max_rfactor: Maximum R-factor.
        min_size: Minimum peak size.
        min_separation: Minimum separation between peaks.
        threshold: Threshold value.
        peak_shape: Peak shape ('L', 'G', etc.).
        max_peaks: Maximum number of peaks.
        mask_file: Optional mask file path.
        threshold_ratio: Threshold ratio. `None` uses the executable default.
        smooth: Whether to apply smoothing.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code, command_string).
    """
    executables = _find_executables()
    
    # Create output filename (remove .h5 extension for consistent naming)
    input_path = Path(input_image)
    if input_path.suffix == '.h5':
        base_name = input_path.stem  # Remove .h5
    else:
        base_name = input_path.stem
    output_file = Path(output_dir) / f'peaks_{base_name}.txt'
    
    # Map peak_shape to single letter if needed
    peak_shape_flag = peak_shape[0].upper() if peak_shape else 'L'
    
    # Build peaksearch command
    cmd = [
        executables['peaksearch'],
        '-b', str(boxsize),
        '-R', str(max_rfactor),
        '-m', str(min_size),
        '-s', str(min_separation),
        '-p', peak_shape_flag,
        '-M', str(max_peaks)
    ]
    
    # Add optional parameters
    # Only add threshold if it's set (not None)
    if threshold is not None:
        cmd.extend(['-t', str(threshold)])
    if mask_file:
        cmd.extend(['-K', mask_file])
    if threshold_ratio is not None:
        cmd.extend(['-T', str(threshold_ratio)])
    if smooth:
        cmd.extend(['-S'])  # Note: -S doesn't take a parameter
    
    # Add input and output files
    cmd.extend([input_image, str(output_file)])
    
    return _run_command(cmd, timeout)


def _run_p2q(peaks_file: str, output_dir: str, geo_file: str, crystal_file: str,
             timeout: int = 300) -> Tuple[bool, str, str, int, str]:
    """
    Run the pix2qs (p2q) executable.
    
    Args:
        peaks_file: Path to peaks file from peaksearch.
        output_dir: Directory for output files.
        geo_file: Path to geometry file.
        crystal_file: Path to crystal file.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code, command_string).
    """
    executables = _find_executables()
    
    # Create output filename
    peaks_path = Path(peaks_file)
    output_file = Path(output_dir) / f'p2q_{peaks_path.stem.replace("peaks_", "")}.txt'
    
    cmd = [
        executables['pix2qs'],
        '-g', geo_file,
        '-x', crystal_file,
        peaks_file,
        str(output_file)
    ]
    return _run_command(cmd, timeout)


def _run_indexing(p2q_file: str, output_dir: str,
                  index_kev_max_calc: float, index_kev_max_test: float,
                  index_angle_tolerance: float, index_cone: float,
                  index_h: int, index_k: int, index_l: int,
                  timeout: int = 300) -> Tuple[bool, str, str, int, str]:
    """
    Run the euler (indexing) executable.
    
    Args:
        p2q_file: Path to p2q file.
        output_dir: Directory for output files.
        index_kev_max_calc: Maximum keV for calculation.
        index_kev_max_test: Maximum keV for testing.
        index_angle_tolerance: Angle tolerance for indexing.
        index_cone: Cone angle for indexing.
        index_h: H index.
        index_k: K index.
        index_l: L index.
        timeout: Command timeout in seconds.
        
    Returns:
        Tuple of (success, stdout, stderr, return_code, command_string).
    """
    executables = _find_executables()
    
    # Create output filename
    p2q_path = Path(p2q_file)
    output_file = Path(output_dir) / f'index_{p2q_path.stem.replace("p2q_", "")}.txt'
    
    cmd = [
        executables['euler'],
        '-q',  # quiet mode
        '-k', str(index_kev_max_calc),
        '-t', str(index_kev_max_test),
        '-a', str(index_angle_tolerance),
        '-c', str(index_cone),
        '-f', p2q_file,
        '-h', str(index_h), str(index_k), str(index_l),
        '-o', str(output_file)
    ]
    return _run_command(cmd, timeout)


def _parse_peaks_output(peaks_file: str) -> int:
    """
    Parse peaks file to count number of peaks found.

    Strategy:
    - Prefer $peakList dims if present: "$peakList <params_per_peak> <n_peaks> ..."
      e.g., "$peakList 8 13" means 13 peaks; "$peakList 8 0" means zero peaks.
    - Otherwise, use explicit $Npeaks header if present.
    - Otherwise, count numeric data rows following the $peakList header
      until the next header/comment/blank line.
    - Robust to files that use $-headers and // comments (not just #).
    """
    try:
        with open(peaks_file, "r") as f:
            lines = f.read().splitlines()

        # 1) Prefer $peakList dims: "$peakList <params_per_peak> <n_peaks> ..."
        for line in lines:
            m = re.match(r'^\s*\$peakList\s+(\d+)\s+(\d+)\b', line)
            if m:
                return int(m.group(2))

        # 2) Next prefer explicit header: $Npeaks <int>
        for line in lines:
            m = re.match(r'^\s*\$Npeaks\s+(\d+)', line)
            if m:
                return int(m.group(1))

        # 3) Fallback: count numeric rows after $peakList
        in_table = False
        count = 0
        for line in lines:
            s = line.strip()
            if not s:
                if in_table:
                    break
                continue
            if s.startswith("$peakList"):
                in_table = True
                continue
            if in_table:
                # Stop if we hit another header/comment
                if s.startswith("$") or s.startswith("#") or s.startswith("//"):
                    break
                # Count if the first token is numeric
                parts = s.split()
                try:
                    float(parts[0])
                    count += 1
                except (ValueError, IndexError):
                    # Non-numeric row signals end or malformed table
                    break
        return count
    except (FileNotFoundError, IOError):
        return 0


def _parse_indexing_output(index_file: str) -> int:
    """
    Parse indexing output file to count number of indexed reflections.

    Only uses the $Nindexed tag (format: "$Nindexed  <int>"). If absent or file unreadable, returns 0.
    """
    try:
        with open(index_file, "r") as f:
            for line in f:
                m = re.match(r'^\s*\$Nindexed\s+(\d+)\b', line)
                if m:
                    return int(m.group(1))
        return 0
    except (FileNotFoundError, IOError):
        return 0


def _parse_npatterns_found(index_file: str) -> int:
    """
    Parse indexing output file to get number of patterns found.

    Only uses the $NpatternsFound tag. If absent or file unreadable, returns 0.
    """
    try:
        with open(index_file, "r") as f:
            for line in f:
                m = re.match(r'^\s*\$NpatternsFound\s+(\d+)\b', line)
                if m:
                    return int(m.group(1))
        return 0
    except (FileNotFoundError, IOError):
        return 0


def _override_depth_in_peaks(peaks_file: str, depth_value: float) -> None:
    """
    Override or insert the $depth header line in a peaks file so downstream pixels2qs
    uses this value instead of what's in the HDF5.
    """
    try:
        with open(peaks_file, "r") as f:
            lines = f.read().splitlines()
    except Exception as e:
        raise RuntimeError(f"Failed to read peaks file for depth override: {e}")
    replaced = False
    # Replace existing $depth if present before the data table
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("$Npeaks") or stripped.startswith("$peakList"):
            break
        if stripped.startswith("$depth"):
            lines[i] = f"$depth\t\t{depth_value}\t\t// depth for depth resolved images (micron)"
            replaced = True
            break
    if not replaced:
        # Insert before the first header separator line starting with //
        insert_at = None
        for i, line in enumerate(lines):
            if line.strip().startswith("//"):
                insert_at = i
                break
        if insert_at is None:
            insert_at = 0
        lines.insert(insert_at, f"$depth\t\t{depth_value}\t\t// depth for depth resolved images (micron)")
    try:
        with open(peaks_file, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write peaks file for depth override: {e}")


def lauego(input_image: str, output_dir: str, geo_file: str, crystal_file: str,
          *,
          # Peak search parameters
          boxsize: int = 5,
          max_rfactor: float = 2.0,
          min_size: int = 3,
          min_separation: int = 10,
          threshold: int = 100,
          peak_shape: str = 'L',
          max_peaks: int = 50,
          mask_file: Optional[str] = None,
          threshold_ratio: Optional[float] = None,
          smooth: bool = False,
          # Indexing parameters
          index_kev_max_calc: float = 30.0,
          index_kev_max_test: float = 35.0,
          index_angle_tolerance: float = 0.12,
          index_cone: float = 72.0,
          index_h: int = 0,
          index_k: int = 0,
          index_l: int = 1,
          # General parameters
          depth_override: Optional[float] = None,
          cosmic_filter: bool = False,
          generate_xml: bool = True,
          xml_output_file: Optional[str] = None,
          timeout: int = 300) -> IndexingResult:
    """Run the LaueGo file- and subprocess-based indexing pipeline.

    Parameters
    ----------
    input_image
        Path to the input diffraction image.
    output_dir
        Root directory for intermediate and final output files.
    geo_file
        Path to the geometry file.
    crystal_file
        Path to the crystal structure file.
    boxsize
        Peak-search fitting box size in pixels.
    max_rfactor
        Maximum peak-fit residual factor.
    min_size
        Minimum peak size in pixels.
    min_separation
        Minimum separation between peaks in pixels.
    threshold
        Intensity threshold for peak detection.
    peak_shape
        Peak model: ``"L"`` for Lorentzian or ``"G"`` for Gaussian.
    max_peaks
        Maximum number of peaks to report.
    mask_file
        Optional path to a peak-search mask file.
    threshold_ratio
        Automatic threshold ratio. `None` uses the LaueGo executable default
        of ``4.0``.
    smooth
        Apply the LaueGo peak-search smoothing option.
    index_kev_max_calc
        Maximum photon energy in keV for reflection calculation.
    index_kev_max_test
        Maximum photon energy in keV for reflection testing.
    index_angle_tolerance
        Angular indexing tolerance in degrees.
    index_cone
        Indexing cone angle in degrees.
    index_h, index_k, index_l
        Components of the preferred Miller-index direction.
    depth_override
        Optional depth in microns written into the intermediate peaks file.
    cosmic_filter
        Must be ``False``. Cosmic-ray filtering is unsupported for indexing;
        ``True`` raises :class:`ValueError` before any output is created.
    generate_xml
        Generate parsed XML output after the subprocess stages.
    xml_output_file
        Explicit XML destination. By default a filename is derived beneath
        ``output_dir``.
    timeout
        Timeout in seconds for each subprocess.

    Returns
    -------
    lauelab.indexing.IndexingResult
        Status, output paths, counts, parsed LaueGo data, logs, and any error.

    Notes
    -----
    This compatibility API writes per-frame intermediate files and invokes the
    ``peaksearch``, ``pix2qs``, and ``euler`` executables. New code should use
    :func:`index_frame` or :class:`Indexer`. ``index`` is an alias for this
    function.
    """
    
    if cosmic_filter:
        raise ValueError("cosmic_filter=True is unsupported for indexing; use False")
    if threshold_ratio is not None and threshold_ratio <= 0:
        raise ValueError("threshold_ratio must be positive or None")
    resolved_threshold_ratio = 4.0 if threshold_ratio is None else threshold_ratio
    recorded_threshold_ratio = resolved_threshold_ratio if threshold is None else None

    # Set up output directories
    try:
        subdirs = _setup_output_dirs(output_dir)
    except Exception as e:
        return IndexingResult(
            success=False,
            output_files={},
            n_peaks_found=0,
            n_indexed=0,
            n_patterns_found=0,
            indexing_data=None,
            step_data=None,
            xml_file=None,
            log="",
            error=f"Failed to create output directories: {e}",
            command_history=()
        )
    
    command_history = []
    log_parts = []
    output_files = {}
    # Determine the base name for all expected outputs from this input
    input_base = Path(input_image).stem
    n_patterns_found = 0
    
    # Step 1: Run peaksearch
    log_parts.append("Running peak search...")
    success, stdout, stderr, returncode, cmd_str = _run_peaksearch(
        input_image, str(subdirs['peaks']),
        boxsize, max_rfactor, min_size,
        min_separation, threshold, peak_shape,
        max_peaks, mask_file,
        threshold_ratio, smooth,
        timeout
    )
    command_history.append(cmd_str)
    log_parts.append(f"Peak search stdout: {stdout}")
    if stderr:
        log_parts.append(f"Peak search stderr: {stderr}")
    
    # Continue even if peaksearch has errors (might still produce output)
    if not success:
        log_parts.append(f"Peak search had errors (return code {returncode}), but checking for output files...")
    
    # Determine expected peaks output file for this image
    expected_peaks = subdirs['peaks'] / f'peaks_{input_base}.txt'
    if expected_peaks.is_file():
        peaks_file = str(expected_peaks)
    else:
        globbed = list(subdirs['peaks'].glob('peaks_*.txt'))
        if len(globbed) == 0:
            return IndexingResult(
                success=False,
                output_files=output_files,
                n_peaks_found=0,
                n_indexed=0,
                n_patterns_found=0,
                indexing_data=None,
                step_data=None,
                xml_file=None,
                log="\n".join(log_parts),
                error="No peaks output file found - peak search completely failed",
                command_history=tuple(command_history)
            )
        return IndexingResult(
            success=False,
            output_files=output_files,
            n_peaks_found=0,
            n_indexed=0,
            n_patterns_found=0,
            indexing_data=None,
            step_data=None,
            xml_file=None,
            log="\n".join(log_parts),
            error=f"No specific peaks file found for this image: expected {expected_peaks.name}",
            command_history=tuple(command_history)
        )
    output_files['peaks'] = peaks_file
    n_peaks_found = _parse_peaks_output(peaks_file)
    # Optional: override depth header before pixels2qs step
    if depth_override is not None:
        try:
            _override_depth_in_peaks(peaks_file, depth_override)
            log_parts.append(f"Depth override applied: $depth = {depth_override}")
            command_history.append(f"override_depth {peaks_file} -> {depth_override}")
        except Exception as e:
            log_parts.append(f"Depth override failed: {e}")
    
    # Only run p2q if we have peaks
    if n_peaks_found > 0:
        # Step 2: Run p2q conversion
        log_parts.append("Running pixel-to-q conversion...")
        success, stdout, stderr, returncode, cmd_str = _run_p2q(
            peaks_file, str(subdirs['p2q']), geo_file, crystal_file, timeout
        )
        command_history.append(cmd_str)
        log_parts.append(f"P2Q stdout: {stdout}")
        if stderr:
            log_parts.append(f"P2Q stderr: {stderr}")
        
        if not success:
            log_parts.append(f"P2Q conversion failed with return code {returncode}, but continuing...")
            # Don't fail completely, continue without p2q results
            return IndexingResult(
                success=True,  # Still success, just no p2q/indexing
                output_files=output_files,
                n_peaks_found=n_peaks_found,
                n_indexed=0,
                n_patterns_found=0,
                indexing_data=None,
                step_data=None,
                xml_file=None,
                log="\n".join(log_parts),
                error=None,
                command_history=tuple(command_history)
            )
        
        # Determine expected p2q output file for this image
        expected_p2q = subdirs['p2q'] / f'p2q_{input_base}.txt'
        if expected_p2q.is_file():
            p2q_file = str(expected_p2q)
        else:
            log_parts.append(f"Expected p2q file {expected_p2q.name} not found, skipping indexing...")
            return IndexingResult(
                success=True,  # Still success, just no indexing
                output_files=output_files,
                n_peaks_found=n_peaks_found,
                n_indexed=0,
                n_patterns_found=0,
                indexing_data=None,
                step_data=None,
                xml_file=None,
                log="\n".join(log_parts),
                error=None,
                command_history=tuple(command_history)
            )
        
        output_files['p2q'] = p2q_file
        
        # Only run indexing if we have at least 2 peaks
        if n_peaks_found > 1:
            # Step 3: Run indexing
            log_parts.append("Running indexing...")
            success, stdout, stderr, returncode, cmd_str = _run_indexing(
                p2q_file, str(subdirs['index']),
                index_kev_max_calc, index_kev_max_test,
                index_angle_tolerance, index_cone,
                index_h, index_k, index_l,
                timeout
            )
            command_history.append(cmd_str)
            log_parts.append(f"Indexing stdout: {stdout}")
            if stderr:
                log_parts.append(f"Indexing stderr: {stderr}")
            
            if not success:
                log_parts.append(f"Indexing failed with return code {returncode}, but continuing...")
                # Don't fail completely, return what we have
                return IndexingResult(
                    success=True,  # Still success, just no indexing results
                    output_files=output_files,
                    n_peaks_found=n_peaks_found,
                    n_indexed=0,
                    n_patterns_found=0,
                    indexing_data=None,
                    step_data=None,
                    xml_file=None,
                    log="\n".join(log_parts),
                    error=None,
                    command_history=tuple(command_history)
                )
            
            # Determine expected index output file for this image
            expected_index = subdirs['index'] / f'index_{input_base}.txt'
            if expected_index.is_file():
                index_file = str(expected_index)
                output_files['index'] = index_file
                n_indexed = _parse_indexing_output(index_file)
                n_patterns_found = _parse_npatterns_found(index_file)
            else:
                n_indexed = 0
                n_patterns_found = 0
                log_parts.append(f"No indexing output file found for expected file {expected_index.name}")
        else:
            log_parts.append(f"Only {n_peaks_found} peak(s) found, need at least 2 for indexing. Skipping indexing step.")
            n_indexed = 0
            n_patterns_found = 0
    else:
        log_parts.append("No peaks found, skipping p2q and indexing steps.")
        n_indexed = 0
        n_patterns_found = 0
    
    log_parts.append("Indexing completed successfully!")
    
    # Generate XML output if requested
    step_data = None
    indexing_data = None
    xml_file = None
    
    if generate_xml:
        try:
            log_parts.append("Generating XML output...")
            
            # Parse all data into Step dataclass
            step_data = parse_full_step_data(
                input_image=input_image,
                output_files=output_files,
                geo_file=geo_file,
                crystal_file=crystal_file,
                cosmic_filter=cosmic_filter,
                boxsize=boxsize,
                max_rfactor=max_rfactor,
                min_size=min_size,
                min_separation=min_separation,
                threshold=threshold,
                peak_shape=peak_shape,
                max_peaks=max_peaks,
                mask_file=mask_file,
                threshold_ratio=recorded_threshold_ratio,
                index_kev_max_calc=index_kev_max_calc,
                index_kev_max_test=index_kev_max_test,
                index_angle_tolerance=index_angle_tolerance,
                index_cone=index_cone,
                index_h=index_h,
                index_k=index_k,
                index_l=index_l,
            )
            
            # Extract indexing data for convenience
            indexing_data = step_data.indexing
            
            # Determine XML output file path
            if xml_output_file is None:
                xml_file = get_default_xml_filename(output_dir, input_image)
            else:
                xml_file = xml_output_file
            
            # Write XML
            write_step_xml(step_data, xml_file)
            log_parts.append(f"XML output written to: {xml_file}")
            command_history.append(f"write_xml -> {xml_file}")
            
        except Exception as e:
            log_parts.append(f"Warning: XML generation failed: {e}")
            # Don't fail the whole operation if XML generation fails
            xml_file = None
    
    return IndexingResult(
        success=True,
        output_files=output_files,
        n_peaks_found=n_peaks_found,
        n_indexed=n_indexed,
        n_patterns_found=n_patterns_found,
        indexing_data=indexing_data,
        step_data=step_data,
        xml_file=xml_file,
        log="\n".join(log_parts),
        error=None,
        command_history=tuple(command_history)
    )
