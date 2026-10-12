# HDF5 file conventions

Indexing results and [reconstruction catalogs and point files](reconstruction-scan-format.md) use these HDF5 conventions, with the exception for point files described under Units. Detector frames and per-depth reconstruction outputs have the 34-ID-E `entry1` layout, which `lauelab` reads and writes but does not define.

## Root attributes

| Attribute | Type | Meaning |
|---|---|---|
| `format` | string | Format identifier, for example `lauelab-indexing-results` |
| `version` | integer | Layout version of that format |
| `lauelab_version` | string | Package version that wrote the file |
| `created` | string | UTC timestamp in ISO 8601 form |
| `source` | string | Optional. The file this one was converted from |

A reader checks `format` and `version` before it reads anything else.

## Versioning

The integer `version` changes when a dataset is removed, renamed, or given a different meaning. Readers reject unsupported versions.

Datasets and attributes can be added within a version. Readers ignore unrecognized fields. A dataset added within a version is optional in the layout module, so every file of that version remains readable. An absent optional dataset represents an unavailable value. Record the reason for a version change in the layout module and the documentation in the same commit.

## Units

Every dimensioned dataset has a `units` attribute with a symbol from the documentation style: `um`, `nm`, `1/nm`, `deg`, `keV`, `pixel`, `s`. A dimensionless dataset has no `units` attribute. A value whose unit is not stated in the acquisition metadata has `units="unspecified"`; readers must not infer a unit or convert the value. A convention that a unit cannot express is stated in another attribute, for example that reciprocal-lattice rows are `a*`, `b*`, `c*` and include the factor of two pi.

Reconstructed point files are an exception. A point file has the 34-ID-E `entry1` layout of its input, and the datasets added by lauelab have the form of the acquisition values: a single value is a `(1,)` array, text is fixed-length and UTF-8 encoded, units have the spelling of the acquisition files, such as `micron` for µm, and a value that does not apply is not written. A point file therefore has the layout of the `reconstructN_cpu` output that beamline tools read.

## Settings and codes

Store the settings of a run as datasets, so that each dimensioned setting can have a `units` attribute; an HDF5 attribute cannot have attributes of its own. The settings of a reconstruction catalog are datasets under `/parameters`. The run parameters of an indexing results file are attributes of `/run`.

A value that was not given or does not apply is stored as `-1` for an integer, NaN for a floating-point value, and an empty string for text. For a numeric dataset, this value is also the HDF5 fill value. Store a fixed set of codes, such as a status, as an HDF5 enumerated type, so that HDF5 tools display the names.

## Ragged data

Store variable-length records in one flat dataset with an offsets array. The offsets start at zero, end at the total row count, and contain one more entry than the number of records. Record `i` occupies rows `offsets[i]:offsets[i + 1]`. Do not use variable-length datasets or region references for numeric data; they cannot be read as one array.

## Storage

Store datasets uncompressed unless the writer has a compression option. A compressed read takes several times longer, and these files are small compared with the frames from which they are calculated. A writer that appends records uses chunked datasets with an unlimited first dimension, which costs a few milliseconds per read.

## Validation and publication

Each lauelab format has a structural validator in addition to its format marker. `lauelab.indexing.validate_results_file` and `lauelab.reconstruct.validate_scan_file` check required datasets, dtypes, shapes, row counts, and applicable offset arrays using bounded reads.

Write output to the `.partial` path of the destination, then close and validate the file before renaming it with `lauelab.publish_file`. The final path then exists only after validation succeeds. If a write fails in the middle of a record, the partial file is invalid and must be written again.

## Layout definition

Define the complete layout of a format in one module: every dataset path, dtype, shape, units, and fixed attribute. The writer, reader, and converter use that table, and a test compares it with the documented layout. The indexing results layout is `lauelab/_results_layout.py`, and the reconstruction scan layout is `lauelab/reconstruct/_scan_layout.py`; the shared root-attribute and version helpers are `lauelab/_hdf5.py`.

The optional indexing source selectors `frames/source_point_ids` and `frames/source_depth_indices` must be present together. In a file with neither, no frame has a scan source. A scan source has a non-empty path and point ID and a nonnegative depth index. An ordinary frame has an empty source point ID and the index `-1`. Validation rejects a file with only one of the two selectors or with inconsistent selector values.
