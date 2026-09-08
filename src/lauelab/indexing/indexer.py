# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""In-process Laue indexing interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Literal, Mapping

import numpy as np

from ._liblaue import Geometry, NativeCrystal, ffi, get_library
from .crystal import Crystal, load_crystal
from .errors import IndexingError, InputError
from ._frame import read_h5_frame, roi_inclusive_end
from .lau_dataclasses.atom import Atom as StepAtom
from .lau_dataclasses.hkls import HKLs
from .lau_dataclasses.indexing import Indexing
from .lau_dataclasses.pattern import Pattern as StepPattern
from .lau_dataclasses.recipLattice import RecipLattice
from .lau_dataclasses.step import Step
from .lau_dataclasses.xtl import Xtl
from .xml_utils import write_combined_xml, write_step_xml


def _raise_native_error(status: int, stage: str, message: str) -> None:
    detail = f"{stage} failed: {message}"
    if status == 2:
        raise MemoryError(detail)
    if status == 1:
        raise InputError(detail)
    raise IndexingError(detail)


PEAK_DTYPE = np.dtype([
    ("fit_x", np.float64),
    ("fit_y", np.float64),
    ("intens", np.float64),
    ("integral", np.float64),
    ("hwhm_x", np.float64),
    ("hwhm_y", np.float64),
    ("tilt", np.float64),
    ("chisq", np.float64),
    ("background", np.float64),
    ("qhat", np.float64, (3,)),
])


@dataclass(frozen=True)
class PeakParams:
    """Peak-search configuration.

    Parameters
    ----------
    boxsize
        Half-width, in pixels, of the square region used to fit each peak.
    max_rfactor
        Maximum fit residual factor accepted for a peak. Must be positive.
    min_size
        Minimum peak size in pixels. Must be positive.
    min_separation
        Minimum separation between accepted peaks in pixels. Must be positive.
    threshold
        Absolute intensity threshold. If `None`, derive a threshold from the
        image statistics and ``threshold_ratio``.
    threshold_ratio
        Scale applied to the image standard deviation when deriving an
        automatic threshold. `None` uses the native default of ``4.0``.
    peak_shape
        Peak model, either ``"Lorentzian"`` or ``"Gaussian"``.
    max_peaks
        Maximum number of peaks returned from one frame. Must be positive.
    smooth
        Whether to smooth the image before peak detection and fitting.

    Notes
    -----
    Instances are immutable. Use :func:`dataclasses.replace` to derive a
    configuration with changed values. Parameter validation occurs when an
    :class:`Indexer` is constructed.
    """

    boxsize: int = 5
    max_rfactor: float = 2.0
    min_size: int = 3
    min_separation: int = 10
    threshold: float | None = 100.0
    threshold_ratio: float | None = None
    peak_shape: Literal["Lorentzian", "Gaussian"] = "Lorentzian"
    max_peaks: int = 50
    smooth: bool = False


@dataclass(frozen=True)
class FrameMetadata:
    """Optional experiment provenance attached to a frame result.

    Parameters
    ----------
    title
        Experiment or scan title.
    sample_name
        Sample name.
    user_name
        User name recorded with the acquisition.
    beamline
        Beamline identifier.
    scan_number
        Scan number.
    date_exposed
        Exposure date or timestamp as recorded by the source file.
    beam_bad
        Source beam-quality flag.
    ccd_shutter
        CCD shutter state.
    light_on
        Illumination-state flag.
    mono_mode
        Monochromator mode.
    sample_position
        Sample ``(x, y, z)`` position in the acquisition coordinate system.
    energy_kev
        Incident energy in keV.
    hutch_temperature
        Hutch temperature in the units supplied by the acquisition metadata.
    sample_distance
        Sample distance in the units supplied by the acquisition metadata.
    detector_id
        Detector identifier expected to match the selected geometry detector.
    exposure_seconds
        Detector exposure time in seconds.

    Notes
    -----
    Instances are immutable. Values supplied explicitly to
    :meth:`Indexer.index` override metadata read from an HDF5 frame.
    """

    title: str | None = None
    sample_name: str | None = None
    user_name: str | None = None
    beamline: str | None = None
    scan_number: int | None = None
    date_exposed: str | None = None
    beam_bad: int | None = None
    ccd_shutter: str | None = None
    light_on: int | None = None
    mono_mode: str | None = None
    sample_position: tuple[float, float, float] | None = None
    energy_kev: float | None = None
    hutch_temperature: float | None = None
    sample_distance: float | None = None
    detector_id: str | None = None
    exposure_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the metadata fields whose values are not `None`.

        Returns
        -------
        dict[str, object]
            A new mapping containing the populated fields.
        """
        return {
            name: value for name, value in self.__dict__.items() if value is not None
        }


@dataclass(frozen=True)
class Pattern:
    """One crystal orientation identified in a diffraction frame.

    Parameters
    ----------
    euler_deg
        Euler angles in degrees, with shape ``(3,)``.
    rotation
        Orientation rotation matrix with shape ``(3, 3)``.
    reciprocal
        Reciprocal-lattice matrix with shape ``(3, 3)`` in ``1/nm``. Rows are
        ``a*``, ``b*``, and ``c*``; a Miller-index row maps to reciprocal
        space as ``q = hkl @ reciprocal``.
    goodness
        Native indexer's goodness score for the pattern.
    rms_error_deg
        Root-mean-square angular indexing error in degrees.
    hkl
        Integer Miller indices with shape ``(n, 3)``.
    pk_index
        Zero-based indices into the frame's peak array, with shape ``(n,)``.
    err_deg
        Per-peak angular errors in degrees, with shape ``(n,)``.
    energy_kev
        Per-peak photon energies in keV, with shape ``(n,)``.
    pred_intens
        Per-peak predicted intensities, with shape ``(n,)``.

    Notes
    -----
    The dataclass is frozen, but its NumPy arrays remain mutable. Each pattern
    owns Python arrays copied from the native result.
    """

    euler_deg: np.ndarray
    rotation: np.ndarray
    reciprocal: np.ndarray
    goodness: float
    rms_error_deg: float
    hkl: np.ndarray
    pk_index: np.ndarray
    err_deg: np.ndarray
    energy_kev: np.ndarray
    pred_intens: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Pattern(n_indexed={self.n_indexed}, goodness={self.goodness:.6g}, "
            f"rms_error_deg={self.rms_error_deg:.6g})"
        )

    @property
    def n_indexed(self) -> int:
        """Number of peaks assigned to this pattern."""
        return len(self.pk_index)


@dataclass(frozen=True)
class IndexParams:
    """Crystal-orientation indexing configuration.

    Parameters
    ----------
    kev_max_calc
        Maximum photon energy in keV used to calculate candidate reflections.
        Must be positive.
    kev_max_test
        Maximum photon energy in keV used to test candidate reflections. Must
        be positive.
    angle_tolerance_deg
        Angular matching tolerance in degrees. Must be positive.
    cone_deg
        Search cone angle in degrees. Must be positive.
    hkl_prefer
        Preferred Miller-index direction as exactly three integers.
    max_data
        Maximum number of detected peaks supplied to the orientation indexer.
        Must be at least two.

    Notes
    -----
    Instances are immutable. Use :func:`dataclasses.replace` to derive a
    configuration with changed values. Parameter validation occurs when an
    :class:`Indexer` is constructed.
    """

    kev_max_calc: float = 30.0
    kev_max_test: float = 35.0
    angle_tolerance_deg: float = 0.12
    cone_deg: float = 72.0
    hkl_prefer: tuple[int, int, int] = (0, 0, 1)
    max_data: int = 250


@dataclass(frozen=True)
class FrameResult:
    """Self-contained result of processing one diffraction frame.

    Parameters
    ----------
    peaks
        Structured peak array with shape ``(n,)``. Fields are ``fit_x``,
        ``fit_y``, ``intens``, ``integral``, ``hwhm_x``, ``hwhm_y``, ``tilt``,
        ``chisq``, ``background``, and the three-component ``qhat`` vector.
    patterns
        Crystal orientations identified in the frame.
    threshold_used
        Intensity threshold used by peak search. This is ``NaN`` when automatic
        thresholding receives no unmasked nonzero pixels.
    total_sum
        Sum of unmasked raw frame pixel values.
    sum_above_threshold
        Sum of pixel values above ``threshold_used``.
    num_above_threshold
        Number of pixels above ``threshold_used``.
    peak_minwidth, peak_maxwidth, peak_max_cent_to_fit, peak_boxsize
        Effective fitting parameters configured by the native peak search.
    peaksearch_seconds
        Elapsed peak-search time in seconds.
    indexing_seconds
        Elapsed orientation-indexing time in seconds. Pixel-to-q conversion is
        not included in either timing field.
    threshold_ratio
        Resolved automatic-threshold ratio supplied to native peak search, or
        ``NaN`` when an absolute threshold made the ratio inactive.
    metadata
        Experiment metadata copied into the result.
    input_image
        Source HDF5 path, or `None` for an in-memory frame.
    image_shape
        Frame shape as ``(rows, columns)``.
    start
        Zero-based detector ``(x, y)`` origin of the frame.
    group
        Detector-pixel grouping factors as ``(x, y)``.
    depth
        Optional sample depth in micrometres passed to the geometry conversion.
    image
        Retained contiguous ``uint16`` frame, or `None` when image retention
        was disabled. This array can alias a contiguous array supplied by the
        caller. Native smoothing uses a separate working copy.

    Notes
    -----
    Native result memory is released before this object is returned. The
    dataclass is frozen, but contained arrays and the metadata mapping may
    remain mutable. A result with no patterns is valid and has ``indexed`` set
    to `False`.
    """

    peaks: np.ndarray
    patterns: tuple[Pattern, ...]
    threshold_used: float
    total_sum: float
    sum_above_threshold: float
    num_above_threshold: int
    peaksearch_seconds: float
    indexing_seconds: float
    threshold_ratio: float = 4.0
    peak_minwidth: float = 0.0
    peak_maxwidth: float = 0.0
    peak_max_cent_to_fit: float = 0.0
    peak_boxsize: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    input_image: str | None = None
    image_shape: tuple[int, int] = (0, 0)
    start: tuple[int, int] = (0, 0)
    group: tuple[int, int] = (1, 1)
    depth: float | None = None
    image: np.ndarray | None = field(default=None, repr=False, compare=False)
    _step: Step | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            f"FrameResult(n_peaks={self.n_peaks}, n_patterns={self.n_patterns}, "
            f"indexed={self.indexed}, image_shape={self.image_shape})"
        )

    @property
    def indexed(self) -> bool:
        """Whether at least one crystal pattern was identified."""
        return bool(self.patterns)

    @property
    def n_peaks(self) -> int:
        """Number of detected peaks."""
        return len(self.peaks)

    @property
    def n_indexed(self) -> int:
        """Total peak assignments across all identified patterns."""
        return sum(pattern.n_indexed for pattern in self.patterns)

    @property
    def n_patterns(self) -> int:
        """Number of identified crystal patterns."""
        return len(self.patterns)

    @property
    def elapsed_seconds(self) -> float:
        """Sum of the recorded peak-search and indexing times in seconds."""
        return self.peaksearch_seconds + self.indexing_seconds

    @property
    def indexed_peak_indices(self) -> np.ndarray:
        """Sorted unique peak indices assigned to any pattern.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of zero-based indices into ``peaks``.
        """
        if not self.patterns:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate([pattern.pk_index for pattern in self.patterns]))

    @property
    def unindexed_peak_indices(self) -> np.ndarray:
        """Sorted peak indices not assigned to a pattern.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of zero-based indices into ``peaks``.
        """
        return np.setdiff1d(np.arange(self.n_peaks), self.indexed_peak_indices)

    def to_step(self) -> Step:
        """Return a copy of the internal LaueGo XML snapshot.

        Returns
        -------
        Step
            A deep copy of the serialization model captured when this result
            was created.

        Raises
        ------
        RuntimeError
            If the result was constructed manually without an XML snapshot.

        Notes
        -----
        ``Step`` is a compatibility representation used for LaueGo XML output,
        not the preferred result API.
        """
        if self._step is None:
            raise RuntimeError("FrameResult has no XML step snapshot")
        return deepcopy(self._step)

    def write_xml(self, path: str | Path) -> None:
        """Write this result in the LaueGo XML format.

        Parameters
        ----------
        path
            Destination XML file. An existing file is replaced.

        Raises
        ------
        RuntimeError
            If the result was constructed manually without an XML snapshot.
        OSError
            If the destination cannot be written.
        """
        write_step_xml(self.to_step(), str(path))


def index_frame(
    frame: np.ndarray | str | Path,
    *,
    geometry: str | Path | Geometry,
    crystal: str | Path | Crystal | None = None,
    peak_params: PeakParams | None = None,
    index_params: IndexParams | None = None,
    detector_index: int = 0,
    detector_id: str | None = None,
    cosmic_filter: bool = False,
    start: tuple[int, int] = (0, 0),
    group: tuple[int, int] = (1, 1),
    depth: float | None = None,
    mask: np.ndarray | None = None,
    metadata: FrameMetadata | Mapping[str, object] | None = None,
    keep_image: bool = True,
) -> FrameResult:
    """Index one frame with a temporary :class:`Indexer`.

    Parameters
    ----------
    frame
        Two-dimensional ``uint16`` NumPy array or path to a supported HDF5
        frame.
    geometry
        Parsed detector geometry or path to a geometry XML file.
    crystal
        Crystal description or crystal XML path. If `None`, peak search and
        pixel-to-q conversion run without orientation indexing.
    peak_params
        Peak-search configuration. Defaults to :class:`PeakParams`.
    index_params
        Orientation-indexing configuration. Defaults to :class:`IndexParams`.
    detector_index
        Physical detector slot in the geometry. This is not the ordinal
        position among active detectors.
    detector_id
        Detector identifier to select instead of ``detector_index``.
    cosmic_filter
        Value recorded in XML output for cosmic-ray filtering provenance.
        This option does not apply an additional Python-side filter.
    start, group
        Full-detector ROI origin and pixel grouping in ``(x, y)`` order.
    depth
        Sample depth in micrometres, or `None` for the beamline origin.
    mask
        Optional mask matching the frame shape. Nonzero pixels are excluded.
    metadata
        Optional frame metadata object or mapping.
    keep_image
        Retain the contiguous input image in the returned result.

    Returns
    -------
    FrameResult
        A self-contained frame result. The input image is retained by default.

    Raises
    ------
    InputError
        If configuration, detector selection, or frame input is invalid.
    MemoryError
        If a native stage cannot allocate required memory.
    IndexingError
        If a native numerical or internal indexing stage fails.

    Notes
    -----
    Construct :class:`Indexer` directly when processing multiple frames so
    parsed geometry and crystal state can be reused.
    """
    return Indexer(
        geometry,
        crystal,
        peak_params=peak_params,
        index_params=index_params,
        detector_index=detector_index,
        detector_id=detector_id,
        cosmic_filter=cosmic_filter,
    ).index(
        frame,
        start=start,
        group=group,
        depth=depth,
        mask=mask,
        metadata=metadata,
        keep_image=keep_image,
    )


class Indexer:
    """Reusable in-process Laue frame indexer.

    Peak search, pixel-to-q conversion, and crystal indexing run through the
    native library without subprocesses or per-frame intermediate text files.

    Parameters
    ----------
    geometry
        Parsed detector geometry or path to a geometry XML file.
    crystal
        Crystal description or crystal XML path. If `None`, frames are peak
        searched and converted to q space without orientation indexing.
    peak_params
        Peak-search configuration. Defaults to :class:`PeakParams`.
    index_params
        Orientation-indexing configuration. Defaults to :class:`IndexParams`.
    detector_index
        Physical detector slot in the geometry. This is not the ordinal
        position among active detectors.
    detector_id
        Detector identifier to select instead of ``detector_index``.
    cosmic_filter
        Value recorded in XML output for cosmic-ray filtering provenance.
        This option does not apply an additional Python-side filter.

    Raises
    ------
    InputError
        If parameters or detector selection are invalid.
    ValueError
        If a geometry or crystal description is invalid.
    ImportError
        If the native indexing library is unavailable.

    Notes
    -----
    Reuse one instance for frames that share geometry, crystal, detector, and
    processing parameters. Calls to :meth:`index` on one instance are safe from
    concurrent Python threads. Each call owns its result storage, and native
    diagnostic output uses thread-local state.

    Use :meth:`replace` to create a separately validated instance with changed
    configuration.
    """

    def __init__(
        self,
        geometry: str | Path | Geometry,
        crystal: str | Path | Crystal | None = None,
        *,
        peak_params: PeakParams | None = None,
        index_params: IndexParams | None = None,
        detector_index: int = 0,
        detector_id: str | None = None,
        cosmic_filter: bool = False,
    ):
        self.geometry_path = geometry.path if isinstance(geometry, Geometry) else Path(geometry)
        self.geometry = geometry if isinstance(geometry, Geometry) else Geometry(self.geometry_path)
        self.crystal = load_crystal(crystal) if isinstance(crystal, (str, Path)) else crystal
        self._crystal = NativeCrystal.create(self.crystal) if self.crystal is not None else None
        self.peak_params = peak_params or PeakParams()
        self.index_params = index_params or IndexParams()
        self._validate_params()
        if len(self.index_params.hkl_prefer) != 3:
            raise InputError("hkl_prefer must contain exactly three integers")
        if self.index_params.max_data < 2:
            raise InputError("max_data must be at least 2")
        if detector_id is not None:
            detector_index = self.geometry.find_detector(detector_id)
            if detector_index < 0:
                raise InputError(f"detector_id {detector_id!r} is not present in the geometry")
        try:
            self.detector = self.geometry.detector(detector_index)
        except (TypeError, OverflowError, ValueError) as error:
            raise InputError(f"detector_index {detector_index!r} is not an active detector slot") from error
        self.detector_index = detector_index
        self.detector_id = self.detector.detector_id
        self.cosmic_filter = cosmic_filter
        self._xtl = self._crystal_to_xtl(self.crystal) if self.crystal else Xtl()

    def __repr__(self) -> str:
        crystal = repr(self.crystal.name) if self.crystal is not None else "None"
        return (
            f"Indexer(detector_id={self.detector_id!r}, detector_index={self.detector_index}, "
            f"crystal={crystal}, smooth={self.peak_params.smooth})"
        )

    def _validate_params(self) -> None:
        peak = self.peak_params
        indexing = self.index_params
        if peak.boxsize < 1 or peak.min_size < 1 or peak.min_separation < 1 or peak.max_peaks < 1:
            raise InputError("peak sizes, separation, and max_peaks must be positive")
        if peak.max_rfactor <= 0:
            raise InputError("max_rfactor must be positive")
        if peak.threshold_ratio is not None and peak.threshold_ratio <= 0:
            raise InputError("threshold_ratio must be positive or None")
        if peak.peak_shape not in {"Lorentzian", "Gaussian"}:
            raise InputError("peak_shape must be 'Lorentzian' or 'Gaussian'")
        if min(indexing.kev_max_calc, indexing.kev_max_test, indexing.angle_tolerance_deg, indexing.cone_deg) <= 0:
            raise InputError("indexing energy and angle parameters must be positive")

    def index(
        self,
        frame: np.ndarray | str | Path,
        *,
        start: tuple[int, int] = (0, 0),
        group: tuple[int, int] = (1, 1),
        depth: float | None = None,
        mask: np.ndarray | None = None,
        metadata: FrameMetadata | Mapping[str, object] | None = None,
        keep_image: bool = True,
    ) -> FrameResult:
        """Process one NumPy frame or HDF5 file without subprocesses.

        Parameters
        ----------
        frame
            Two-dimensional ``uint16`` NumPy array or path to a supported HDF5
            frame.
        start
            Zero-based detector ``(x, y)`` origin for an in-memory frame.
        group
            Positive detector-pixel grouping factors as ``(x, y)``.
        depth
            Optional finite sample depth in micrometres passed to pixel-to-q
            conversion.
        mask
            Array with the same shape as ``frame``. Values are converted to a
            boolean ``uint8`` mask and passed to native peak search, where
            nonzero pixels are masked.
        metadata
            Experiment metadata for XML output. Explicit values override
            metadata loaded from an HDF5 file.
        keep_image
            Retain the contiguous input image in the returned result's
            ``image`` attribute.

        Returns
        -------
        FrameResult
            A self-contained result whose native backing allocations have
            already been released.

        Raises
        ------
        InputError
            If the frame, region, mask, metadata, or selected detector is
            invalid.
        KeyError
            If a required HDF5 image dataset is missing.
        OSError
            If an HDF5 input file cannot be opened.
        MemoryError
            If a native stage cannot allocate required memory.
        IndexingError
            If a native numerical or internal stage fails.

        Notes
        -----
        For HDF5 input, detector ``start`` and ``group`` values from the file
        take precedence over method arguments when present. A retained image
        can alias a contiguous array supplied by the caller. Native peak search
        uses a separate working copy, so smoothing does not modify that array.

        Frame statistics exclude masked pixels and always describe the raw
        input image. Smoothing applies only to peak detection and fitting. Under
        automatic thresholding, a frame with no
        unmasked nonzero pixels returns a valid empty result with
        ``threshold_used`` set to ``NaN``.

        Native code can write diagnostics to stdout or stderr before Python
        raises an exception.
        """
        input_image = None
        supplied_metadata = metadata.as_dict() if isinstance(metadata, FrameMetadata) else dict(metadata or {})
        if isinstance(frame, (str, Path)):
            input_image = str(frame)
            source, file_metadata, file_processing = read_h5_frame(frame)
            supplied_metadata = {**file_metadata, **supplied_metadata}
            start = file_processing.get("start", start)
            group = file_processing.get("group", group)
        else:
            source = np.asarray(frame)
        if source.ndim != 2 or source.dtype != np.uint16:
            raise InputError(
                f"frame must be a 2D uint16 array; received shape={source.shape}, dtype={source.dtype}"
            )
        image = np.ascontiguousarray(source)
        if (
            len(start) != 2
            or len(group) != 2
            or not all(isinstance(value, (int, np.integer)) for value in (*start, *group))
            or min(start) < 0
            or min(group) < 1
        ):
            raise InputError("start must be two nonnegative integers and group two positive integers")
        start = tuple(int(value) for value in start)
        group = tuple(int(value) for value in group)
        detector_id = supplied_metadata.get("detector_id")
        if detector_id is not None and detector_id != self.detector.detector_id:
            raise InputError(
                f"frame detector_id {detector_id!r} does not match selected detector "
                f"{self.detector.detector_id!r}"
            )
        end_x, end_y = roi_inclusive_end(image.shape, start, group)
        if end_x >= self.detector.nx or end_y >= self.detector.ny:
            raise InputError(
                f"frame ROI start={start}, shape={image.shape}, group={group} exceeds "
                f"detector bounds {self.detector.nx}x{self.detector.ny}"
            )
        if depth is not None and not np.isfinite(depth):
            raise InputError("depth must be finite when provided")

        mask_buffer = None
        mask_pointer = ffi.NULL
        if mask is not None:
            mask_buffer = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8)
            if mask_buffer.shape != image.shape:
                raise InputError(f"mask shape {mask_buffer.shape} does not match frame shape {image.shape}")
            mask_pointer = ffi.from_buffer("unsigned char[]", mask_buffer)

        threshold_ratio = (
            4.0 if self.peak_params.threshold_ratio is None else self.peak_params.threshold_ratio
        )
        recorded_threshold_ratio = (
            threshold_ratio if self.peak_params.threshold is None else np.nan
        )
        params = ffi.new("laue_peak_params *")
        params.boxsize = self.peak_params.boxsize
        params.max_rfactor = self.peak_params.max_rfactor
        params.min_size = self.peak_params.min_size
        params.min_separation = self.peak_params.min_separation
        params.threshold = np.nan if self.peak_params.threshold is None else self.peak_params.threshold
        params.threshold_ratio = threshold_ratio
        params.peak_shape = 1 if self.peak_params.peak_shape == "Gaussian" else 0
        params.max_peaks = self.peak_params.max_peaks
        params.smooth = self.peak_params.smooth
        params.mask = mask_pointer

        library = get_library()
        result = ffi.new("laue_frame_result *")
        pixels = ffi.from_buffer("unsigned short[]", image)
        peaksearch_started = perf_counter()
        status = library.laue_find_peaks(pixels, image.shape[1], image.shape[0], params, result)
        peaksearch_seconds = perf_counter() - peaksearch_started
        if status:
            message = ffi.string(result.message).decode(errors="replace")
            library.laue_frame_result_free(result)
            _raise_native_error(status, "peak search", message)

        try:
            result.startx, result.starty = start
            result.groupx, result.groupy = group
            result.depth = np.nan if depth is None else depth
            status = library.laue_pixels_to_q(self.geometry._handle, self.detector_index, result)
            if status:
                _raise_native_error(
                    status,
                    "pixel-to-q conversion",
                    ffi.string(result.message).decode(errors="replace"),
                )

            indexing_started = perf_counter()
            if self._crystal is not None and result.n_peaks > 1:
                index_params = ffi.new("laue_index_params *")
                index_params.kev_max_calc = self.index_params.kev_max_calc
                index_params.kev_max_test = self.index_params.kev_max_test
                index_params.angle_tolerance_deg = self.index_params.angle_tolerance_deg
                index_params.cone_deg = self.index_params.cone_deg
                for index, value in enumerate(self.index_params.hkl_prefer):
                    index_params.hkl_prefer[index] = value
                index_params.max_data = self.index_params.max_data
                status = library.laue_index(self._crystal._handle, index_params, result)
                if status:
                    _raise_native_error(
                        status,
                        "orientation indexing",
                        ffi.string(result.message).decode(errors="replace"),
                    )

            indexing_seconds = perf_counter() - indexing_started

            peaks = np.empty(result.n_peaks, dtype=PEAK_DTYPE)
            for index in range(result.n_peaks):
                peak = result.peaks[index]
                peaks[index] = (
                    peak.fit_x, peak.fit_y, peak.intens, peak.integral,
                    peak.hwhm_x, peak.hwhm_y, peak.tilt, peak.chisq,
                    peak.background, tuple(peak.qhat),
                )
            patterns = []
            for index in range(result.n_patterns):
                pattern = result.patterns[index]
                count = pattern.n_indexed
                patterns.append(Pattern(
                    euler_deg=np.asarray(list(pattern.euler_deg)),
                    rotation=np.asarray([list(row) for row in pattern.rotation]),
                    reciprocal=np.asarray([list(row) for row in pattern.recip]),
                    goodness=pattern.goodness,
                    rms_error_deg=pattern.rms_error_deg,
                    hkl=np.asarray([pattern.hkl[i] for i in range(3 * count)], dtype=np.int32).reshape((-1, 3)),
                    pk_index=np.asarray([pattern.pk_index[i] for i in range(count)], dtype=np.int32),
                    err_deg=np.asarray([pattern.err_deg[i] for i in range(count)]),
                    energy_kev=np.asarray([pattern.energy_kev[i] for i in range(count)]),
                    pred_intens=np.asarray([pattern.pred_intens[i] for i in range(count)]),
                ))
            frame_result = FrameResult(
                peaks=peaks,
                patterns=tuple(patterns),
                threshold_used=result.threshold_used,
                threshold_ratio=recorded_threshold_ratio,
                total_sum=result.total_sum,
                sum_above_threshold=result.sum_above_threshold,
                num_above_threshold=result.num_above_threshold,
                peak_minwidth=result.peak_minwidth,
                peak_maxwidth=result.peak_maxwidth,
                peak_max_cent_to_fit=result.peak_max_cent_to_fit,
                peak_boxsize=result.peak_boxsize,
                peaksearch_seconds=peaksearch_seconds,
                indexing_seconds=indexing_seconds,
                metadata=supplied_metadata,
                input_image=input_image,
                image_shape=image.shape,
                start=start,
                group=group,
                depth=depth,
                image=image if keep_image else None,
            )
            object.__setattr__(frame_result, "_step", self._to_step(frame_result))
            return frame_result
        finally:
            library.laue_frame_result_free(result)

    def index_many(
        self, frames: Iterable[np.ndarray | str | Path], *, keep_images: bool = False
    ) -> list[FrameResult]:
        """Index frames in order while reusing parsed configuration.

        Parameters
        ----------
        frames
            Iterable of two-dimensional ``uint16`` arrays or supported HDF5
            paths.
        keep_images
            Retain each input image in its result. Defaults to `False` to limit
            batch memory use.

        Returns
        -------
        list[FrameResult]
            Results in the same order as the input iterable.

        Raises
        ------
        InputError
            If a frame or its metadata is invalid.
        MemoryError
            If a native stage cannot allocate required memory.
        IndexingError
            If a native numerical or internal stage fails.

        Notes
        -----
        Frames are processed sequentially. Processing stops on the first
        exception.
        """
        return [self.index(frame, keep_image=keep_images) for frame in frames]

    def replace(self, **changes) -> "Indexer":
        """Create an indexer with selected configuration values replaced.

        Parameters
        ----------
        **changes
            Constructor arguments to replace. Supported names are ``geometry``,
            ``crystal``, ``peak_params``, ``index_params``,
            ``detector_index``, ``detector_id``, and ``cosmic_filter``.

        Returns
        -------
        Indexer
            A new, independently validated indexer.

        Raises
        ------
        TypeError
            If an unknown constructor argument is supplied.
        InputError
            If replacement parameters or detector selection are invalid.

        Notes
        -----
        Native geometry and crystal state is constructed for the new instance;
        it is not shared with the original indexer.
        """
        values = {
            "geometry": self.geometry,
            "crystal": self.crystal,
            "peak_params": self.peak_params,
            "index_params": self.index_params,
            "detector_index": self.detector_index,
            "cosmic_filter": self.cosmic_filter,
        }
        values.update(changes)
        return type(self)(**values)

    def results_writer(self, path: str | Path, **kwargs):
        """Create a streaming HDF5 writer bound to this indexer's configuration."""
        from .results import ResultsWriter

        return ResultsWriter(
            path,
            crystal=self.crystal,
            geometry=self.geometry,
            peak_params=self.peak_params,
            index_params=self.index_params,
            detector_index=self.detector_index,
            detector_id=self.detector_id,
            cosmic_filter=self.cosmic_filter,
            **kwargs,
        )

    def write_results(
        self,
        results: Iterable[FrameResult],
        path: str | Path,
        *,
        frame_ids=None,
        overwrite: bool = False,
    ) -> None:
        """Write an iterable of results to one indexing-results HDF5 file."""
        from .results import write_results

        write_results(
            self, results, path, frame_ids=frame_ids, overwrite=overwrite
        )

    def write_many_xml(self, results: Iterable[FrameResult], path: str | Path) -> None:
        """Write multiple results to one LaueGo XML document.

        Parameters
        ----------
        results
            Frame results written in iteration order.
        path
            Destination XML file. An existing file is replaced.

        Raises
        ------
        RuntimeError
            If any result has no XML snapshot.
        OSError
            If the destination cannot be written.
        """
        write_combined_xml([result.to_step() for result in results], str(path))

    @staticmethod
    def _crystal_to_xtl(crystal: Crystal) -> Xtl:
        cell = crystal.cell
        xtl = Xtl(
            structureDesc=crystal.name,
            xtalFileName=crystal.source,
            SpaceGroup=(
                f"{crystal.space_group}:{crystal.setting}"
                if crystal.setting is not None
                else crystal.space_group
            ),
            latticeParameters=" ".join(str(value) for value in (
                cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma
            )),
            lengthUnit=cell.unit,
        )
        for number, atom in enumerate(crystal.atoms, start=1):
            xtl.atoms.append(StepAtom(
                n=number,
                symbol=atom.symbol,
                label=atom.label or atom.symbol,
                values=" ".join(str(value) for value in atom.position),
                occupancy=atom.occupancy,
            ))
        return xtl

    def _to_step(self, result: FrameResult) -> Step:
        metadata = result.metadata
        step = Step()
        names = {
            "title": "title",
            "sample_name": "sampleName",
            "user_name": "userName",
            "beamline": "beamline",
            "scan_number": "scanNum",
            "date_exposed": "dateExposed",
            "beam_bad": "beamBad",
            "ccd_shutter": "CCDshutter",
            "light_on": "lightOn",
            "mono_mode": "monoMode",
            "energy_kev": "energy",
            "hutch_temperature": "hutchTemperature",
            "sample_distance": "sampleDistance",
        }
        for source, destination in names.items():
            if source in metadata:
                setattr(step, destination, metadata[source])
        if "sample_position" in metadata:
            step.Xsample, step.Ysample, step.Zsample = metadata["sample_position"]
        step.depth = result.depth
        step.detector.inputImage = result.input_image
        step.detector.detectorID = metadata.get("detector_id", self.detector_id)
        step.detector.exposure = metadata.get("exposure_seconds")
        step.detector.Ny, step.detector.Nx = result.image_shape
        step.detector.totalSum = result.total_sum
        step.detector.sumAboveThreshold = result.sum_above_threshold
        step.detector.numAboveThreshold = result.num_above_threshold
        step.detector.cosmicFilter = self.cosmic_filter
        step.detector.geoFile = str(self.geometry_path)
        roi = step.detector.roi
        roi.startx, roi.starty = result.start
        roi.groupx, roi.groupy = result.group
        roi.endx, roi.endy = roi_inclusive_end(
            result.image_shape, result.start, result.group
        )

        peak_data = step.detector.peaksXY
        peak_data.peakProgram = "liblaue"
        peak_data.minwidth = result.peak_minwidth
        peak_data.threshold = result.threshold_used
        peak_data.thresholdRatio = result.threshold_ratio
        peak_data.maxRfactor = self.peak_params.max_rfactor
        peak_data.maxwidth = result.peak_maxwidth
        peak_data.maxCentToFit = result.peak_max_cent_to_fit
        peak_data.boxsize = result.peak_boxsize
        peak_data.NpeakMax = self.peak_params.max_peaks
        peak_data.minSeparation = self.peak_params.min_separation
        peak_data.peakShape = self.peak_params.peak_shape
        peak_data.Npeaks = result.n_peaks
        peak_data.executionTime = result.peaksearch_seconds
        for peak in result.peaks:
            peak_data.addPeak(*[str(peak[name]) for name in (
                "fit_x", "fit_y", "intens", "integral", "hwhm_x", "hwhm_y", "tilt", "chisq"
            )])
            peak_data.background.append(str(peak["background"]))
            peak_data.addQVector(*(str(value) for value in peak["qhat"]))

        indexing = Indexing(
            indexProgram="liblaue",
            Nindexed=result.n_indexed,
            Npeaks=result.n_peaks,
            NpatternsFound=len(result.patterns),
            keVmaxCalc=self.index_params.kev_max_calc,
            keVmaxTest=self.index_params.kev_max_test,
            angleTolerance=self.index_params.angle_tolerance_deg,
            cone=self.index_params.cone_deg,
            hklPrefer=" ".join(str(value) for value in self.index_params.hkl_prefer),
            executionTime=result.indexing_seconds,
            xtl=deepcopy(self._xtl),
        )
        for number, source in enumerate(result.patterns):
            reciprocal = RecipLattice(
                astar=" ".join(str(value) for value in source.reciprocal[0]),
                bstar=" ".join(str(value) for value in source.reciprocal[1]),
                cstar=" ".join(str(value) for value in source.reciprocal[2]),
            )
            hkls = HKLs(
                h=[str(value) for value in source.hkl[:, 0]],
                k=[str(value) for value in source.hkl[:, 1]],
                l=[str(value) for value in source.hkl[:, 2]],
                PkIndex=[str(value) for value in source.pk_index],
                err_deg=[str(value) for value in source.err_deg],
                energy_kev=[str(value) for value in source.energy_kev],
                pred_intens=[str(value) for value in source.pred_intens],
            )
            indexing.patterns.append(StepPattern(
                num=number,
                rms_error=source.rms_error_deg,
                goodness=source.goodness,
                Nindexed=source.n_indexed,
                recip_lattice=reciprocal,
                hkl_s=hkls,
            ))
        step.indexing = indexing
        return step
