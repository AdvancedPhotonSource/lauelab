# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""XML generation utilities for Laue indexing results."""

from typing import List
from xml.etree import ElementTree
from pathlib import Path

from lauelab.indexing.lau_dataclasses.step import Step

_XML_HEADER = '<?xml version="1.0" ?>\n<AllSteps>'
_ROOT_OPEN = "<AllSteps>"
_ROOT_CLOSE = "</AllSteps>"


def _step_chunk(step: Step) -> str:
    """Serialize one step exactly as it appears inside a combined document."""
    root = ElementTree.Element("AllSteps")
    root.append(step.getXMLElem())
    ElementTree.indent(root, space="    ")
    text = ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)
    assert text.startswith(_ROOT_OPEN) and text.endswith(_ROOT_CLOSE)
    # Inner text is "\n    <step>...</step>\n"; the final newline belongs to
    # whichever step is last, so it is written by close().
    return text[len(_ROOT_OPEN):-len(_ROOT_CLOSE)].rstrip("\n")


class XmlResultsWriter:
    """Write frame results to one LaueGo ``AllSteps`` XML document incrementally.

    Parameters
    ----------
    path
        Destination XML file.
    overwrite
        Replace an existing file. The default refuses with ``FileExistsError``.

    Attributes
    ----------
    count : int
        Steps written so far.
    failed : bool
        `True` once a write to the file raised. The document may then be
        truncated or malformed; further appends are refused.
    error : Exception or None
        The exception that set ``failed``.

    Notes
    -----
    Output is byte-identical to ``write_combined_xml`` for the same steps,
    and each step is serialized, written, and flushed when it arrives, so
    memory use does not grow with the number of frames. Use the writer as a
    context manager; leaving the block writes the closing tag and closes the
    file even when the body raised. After a failed write, ``close()`` first
    truncates the file back to the end of the last completely written step,
    so the document stays well formed up to that step whenever the filesystem
    still accepts the closing tag.

    This is the auxiliary output of an indexing run. A caller that also
    writes an HDF5 results file should treat a failure here as a warning about
    the XML alone; the HDF5 file is unaffected.
    """

    def __init__(self, path: str | Path, *, overwrite: bool = False):
        self.path = Path(path)
        self._mode = "w" if overwrite else "x"
        self._handle = None
        self._complete_end = 0
        self.count = 0
        self.error: Exception | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def __enter__(self) -> "XmlResultsWriter":
        self._handle = open(self.path, self._mode, encoding="utf-8", newline="")
        try:
            self._handle.write(_XML_HEADER)
            self._handle.flush()
            self._complete_end = self._handle.tell()
        except Exception as error:
            self.error = error
            self._handle.close()
            self._handle = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def append(self, item) -> None:
        """Append one :class:`~lauelab.indexing.FrameResult` or LaueGo ``Step``.

        Raises
        ------
        RuntimeError
            If the writer is not open, has already failed, or the result has
            no XML snapshot.
        TypeError
            If ``item`` is neither a ``FrameResult`` nor a ``Step``.
        OSError
            If writing fails. The writer is then marked ``failed``.
        """
        if self._handle is None:
            raise RuntimeError("XmlResultsWriter must be used as a context manager")
        if self.failed:
            raise RuntimeError(f"XmlResultsWriter failed earlier ({self.error!r}); {self.path} is not a complete document")
        if isinstance(item, Step):
            step = item
        elif hasattr(item, "to_step"):
            step = item.to_step()
        else:
            raise TypeError("append expects a FrameResult or a Step")
        chunk = _step_chunk(step)
        try:
            self._handle.write(chunk)
            self._handle.flush()
            self._complete_end = self._handle.tell()
        except Exception as error:
            self.error = error
            raise
        self.count += 1

    def close(self) -> None:
        """Write the closing tag and close the file. Safe to call twice.

        After a failed append the file is truncated to the end of the last
        complete step and the closing tag is still attempted, so that the
        document ends properly whenever the filesystem allows; a second
        failure is recorded in ``error`` and not raised again.
        """
        handle, self._handle = self._handle, None
        if handle is None:
            return
        closing = _ROOT_CLOSE if self.count == 0 else "\n" + _ROOT_CLOSE
        already_failed = self.failed
        try:
            if already_failed:
                handle.seek(self._complete_end)
                handle.truncate()
            handle.write(closing)
            handle.close()
        except Exception as error:
            try:
                handle.close()
            except Exception:
                pass
            if already_failed:
                return
            self.error = error
            raise


def write_step_xml(step: Step, xml_file: str) -> None:
    """
    Write a single step to an XML file.
    
    Args:
        step: Step object containing all indexing data.
        xml_file: Path to output XML file.
    """
    root = ElementTree.Element('AllSteps')
    root.append(step.getXMLElem())
    
    ElementTree.indent(root, space='    ')
    xml = ElementTree.tostring(root, encoding='unicode', short_empty_elements=False)
    Path(xml_file).write_text('<?xml version="1.0" ?>\n' + xml, encoding='utf-8')


def write_combined_xml(steps: List[Step], xml_file: str) -> None:
    """
    Write multiple steps to a combined XML file.
    
    Args:
        steps: List of Step objects to write.
        xml_file: Path to output XML file.
    """
    root = ElementTree.Element('AllSteps')
    for step in steps:
        root.append(step.getXMLElem())
    
    ElementTree.indent(root, space='    ')
    xml = ElementTree.tostring(root, encoding='unicode', short_empty_elements=False)
    Path(xml_file).write_text('<?xml version="1.0" ?>\n' + xml, encoding='utf-8')


def get_default_xml_filename(output_dir: str, input_image: str = '', prefix: str = '') -> str:
    """
    Get the default XML output filename based on the input image.
    
    XML files are stored in the 'xml' subdirectory of the output directory,
    following the same pattern as other intermediate files (peaks, p2q, index).
    
    Args:
        output_dir: Output directory path.
        input_image: Path to input image file (used to derive unique filename).
        prefix: Optional prefix for the filename.
        
    Returns:
        Full path to the XML file in the xml subdirectory.
    """
    xml_dir = Path(output_dir) / 'xml'
    
    # Create xml directory if it doesn't exist
    xml_dir.mkdir(parents=True, exist_ok=True)
    
    if input_image:
        # Derive filename from input image, similar to peaks/p2q/index files
        input_base = Path(input_image).stem
        filename = f'{prefix}indexed_{input_base}.xml' if prefix else f'indexed_{input_base}.xml'
    else:
        # Fallback for combined/batch operations
        filename = f'{prefix}indexed.xml' if prefix else 'indexed.xml'
    
    return str(xml_dir / filename)
