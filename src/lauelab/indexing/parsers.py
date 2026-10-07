# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Parsing utilities for Laue indexing output files."""

from typing import Optional
from pathlib import Path

from lauelab.indexing.lau_dataclasses.step import Step
from lauelab.indexing.lau_dataclasses.indexing import Indexing
from lauelab.indexing.lau_dataclasses.peaksXY import PeaksXY


def parse_peaks_file(peaks_file: str, step: Step) -> None:
    """
    Parse peak search output file and populate step data.
    
    Peak search command outputs a txt file in the form:
    $attr1 val1
    $attr2 val2
    ...
    followed by a matrix with peak data.
    
    Args:
        peaks_file: Path to peaks output file.
        step: Step object to populate with peak data.
    """
    with open(peaks_file, encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
        # Reset peaks lists to avoid accumulation across calls/tests
        px = step.detector.peaksXY
        px.Xpixel.clear()
        px.Ypixel.clear()
        px.Intens.clear()
        px.Integral.clear()
        px.hwhmX.clear()
        px.hwhmY.clear()
        px.tilt.clear()
        px.chisq.clear()
        px.Qx.clear()
        px.Qy.clear()
        px.Qz.clear()

        for line in lines:
            line = line.split('\t')
            vals = []
            for val in line:
                if val and not val.startswith('//'):
                    vals.append(val.strip().replace('$', ''))
            if len(vals) == 2:
                step.set(vals[0], vals[1])
            elif vals:
                tokens = vals[0].split()
                # Only treat as peak rows if there are exactly 8 values
                if len(tokens) == 8:
                    step.detector.peaksXY.addPeak(*tokens)


def parse_p2q_file(p2q_file: str, peaks_xy: PeaksXY) -> None:
    """
    Parse pixel-to-q conversion output file and add Q vectors to peaks.
    
    P2Q outputs a txt file with qX qY qZ values listed in a matrix
    part way through the file after the $N_Ghat+Intens marker.
    
    Args:
        p2q_file: Path to p2q output file.
        peaks_xy: PeaksXY object to add Q vectors to.
    """
    with open(p2q_file, encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()

    get_line = False
    for line in lines:
        if get_line:
            line = line.split(', ')[:3]
            peaks_xy.addQVector(*line)
        if '$N_Ghat+Intens' in line:
            # The values we care about are listed after $N_Ghat+Intens
            get_line = True


def parse_indexing_file(index_file: str, n_peaks: int) -> Indexing:
    """
    Parse indexing output file and create Indexing dataclass.
    
    Index command outputs a txt file in the form:
    $attr1 val1
    $attr2 val2
    ...
    followed by pattern data with indexed reflections.
    
    Args:
        index_file: Path to indexing output file.
        n_peaks: Number of peaks found (for metadata).
        
    Returns:
        Indexing object with parsed data.
    """
    indexing = Indexing()
    with open(index_file, encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            vals = []
            for val in line:
                if val and not val.startswith('//'):
                    val = val.strip()
                    if val.startswith('$'):
                        val = val.replace('$', '')
                    vals.append(val)
            if len(vals) == 2:
                indexing.set(vals[0], vals[1])
            elif vals and vals[0].startswith('['):
                indexing.patterns[-1].hkl_s.fromString(vals[0])
    indexing.set('Npeaks', n_peaks)
    return indexing


def parse_full_step_data(
    input_image: str,
    output_files: dict,
    geo_file: str,
    crystal_file: Optional[str] = None,
    cosmic_filter: bool = False,
    boxsize: int = 5,
    max_rfactor: float = 2.0,
    min_size: float = 3,
    min_separation: int = 10,
    threshold: int = 100,
    peak_shape: str = 'L',
    max_peaks: int = 50,
    mask_file: Optional[str] = None,
    threshold_ratio: Optional[float] = None,
    index_kev_max_calc: float = 30.0,
    index_kev_max_test: float = 35.0,
    index_angle_tolerance: float = 0.12,
    index_cone: float = 72.0,
    index_h: int = 0,
    index_k: int = 0,
    index_l: int = 1,
) -> Step:
    """
    Parse all output files and create a complete Step dataclass.
    
    Args:
        input_image: Path to input image file.
        output_files: Dictionary of output file paths (peaks, p2q, index).
        geo_file: Path to geometry file.
        crystal_file: Path to crystal file (optional).
        cosmic_filter: Whether cosmic filter was applied.
        boxsize: Box size used in peak search.
        max_rfactor: Maximum R-factor used.
        min_size: Minimum peak size used.
        min_separation: Minimum separation used.
        threshold: Threshold value used.
        peak_shape: Peak shape used ('L' or 'G').
        max_peaks: Maximum peaks parameter.
        mask_file: Mask file used (if any).
        threshold_ratio: Threshold ratio used.
        index_kev_max_calc: Max keV for calculation.
        index_kev_max_test: Max keV for testing.
        index_angle_tolerance: Angle tolerance used.
        index_cone: Cone angle used.
        index_h: H component of reference vector.
        index_k: K component of reference vector.
        index_l: L component of reference vector.
        
    Returns:
        Complete Step object with all parsed data.
    """
    # Create step and parse H5 metadata
    step = Step()
    step.fromH5(input_image)
    
    # Set detector metadata
    step.detector.set('cosmicFilter', str(cosmic_filter))
    step.detector.set('geoFile', geo_file)
    step.detector.set('inputImage', input_image)
    
    # Parse peaks file if available
    if 'peaks' in output_files:
        peaks_file = output_files['peaks']
        parse_peaks_file(peaks_file, step)
        
        # Set peak search parameters
        step.detector.peaksXY.set('peakProgram', 'peaksearch')
        step.detector.peaksXY.set('boxsize', boxsize)
        step.detector.peaksXY.set('maxRfactor', max_rfactor)
        step.detector.peaksXY.set('threshold', threshold)
        step.detector.peaksXY.set('thresholdRatio', threshold_ratio)
        step.detector.peaksXY.set('peakShape', peak_shape)
        step.detector.peaksXY.NpeakMax = max_peaks
        step.detector.peaksXY.minSeparation = min_separation
        if mask_file:
            step.detector.peaksXY.maskFile = mask_file
    
    # Parse p2q file if available
    n_peaks = step.detector.peaksXY.Npeaks or 0
    if 'p2q' in output_files and n_peaks > 0:
        p2q_file = output_files['p2q']
        parse_p2q_file(p2q_file, step.detector.peaksXY)
    
    # Parse indexing file if available
    if 'index' in output_files and n_peaks > 1:
        index_file = output_files['index']
        step.indexing = parse_indexing_file(index_file, n_peaks)
        step.indexing.set('indexProgram', 'euler')
        step.indexing.set('keVmaxCalc', index_kev_max_calc)
        step.indexing.set('keVmaxTest', index_kev_max_test)
        step.indexing.set('angleTolerance', index_angle_tolerance)
        step.indexing.set('cone', index_cone)
        step.indexing.set('hklPrefer', f'{index_h} {index_k} {index_l}')
    else:
        # No indexing results
        step.indexing = Indexing()
        step.indexing.set('Nindexed', 0)
        step.indexing.set('indexProgram', 'euler')
    
    return step
