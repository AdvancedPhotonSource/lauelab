# HDF5 file conventions

These conventions apply to HDF5 files whose layout `lauelab` defines. The indexing results file and the [reconstruction scan file](reconstruction-scan-format.md) share these conventions. Detector frames and per-depth reconstruction outputs follow the 34-ID-E `entry1` layout, which `lauelab` reads and writes but does not define.

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

Datasets and attributes can be added within the current version. Readers ignore unrecognized fields, and the layout module marks newly added datasets as optional so that earlier files remain readable. An absent optional dataset represents an unavailable value. Record the reason for a version change in the layout module and the documentation in the same commit.

## Units

Every dimensioned dataset carries a `units` attribute using the symbols from the documentation style: `um`, `nm`, `1/nm`, `deg`, `keV`, `pixel`, `s`. A dimensionless dataset has no `units` attribute. A value whose unit the acquisition metadata does not establish is kept with `units="unspecified"`; readers must not infer a unit or convert it. A convention a unit cannot express is a further attribute, for example that reciprocal-lattice rows are `a*`, `b*`, `c*` and include the factor of two pi.

## Ragged data

Store variable-length records in one flat dataset with an offsets array. The offsets start at zero, end at the total row count, and contain one more entry than the number of records. Record `i` occupies rows `offsets[i]:offsets[i + 1]`. Do not use variable-length datasets or region references for numeric data; they cannot be read as one array.

## Storage

Store datasets uncompressed unless the writer offers compression as an option; compressed reads cost several times more and these files are small next to the frames they summarize. A writer that appends records uses chunked datasets with an unlimited first dimension, which costs a few milliseconds per read.

## Validation and publication

Each lauelab-defined format provides structural validation in addition to its format marker. `lauelab.indexing.validate_results_file` and `lauelab.reconstruct.validate_scan_file` check required datasets, dtypes, shapes, row counts, and applicable offset arrays using bounded reads.

Write output to the destination's `.partial` path, then close and validate the file before renaming it with `lauelab.publish_file`. This makes the final path available only after validation succeeds. A mid-record write failure invalidates the partial file and requires a rewrite.

## Layout definition

Define the complete layout of a format in one module: every dataset path, dtype, shape, units, and fixed attribute. The writer, reader, converter, and reference documentation read that one table, and a test compares it against the documented layout. The indexing results layout is `lauelab/_results_layout.py`, and the reconstruction scan layout is `lauelab/reconstruct/_scan_layout.py`; the shared root-attribute and version helpers are `lauelab/_hdf5.py`.

The optional indexing source selectors `frames/source_point_ids` and `frames/source_depth_indices` must appear together. Old files with neither remain readable. A scan source requires a non-empty path and point ID and a nonnegative depth index. An ordinary frame uses an empty source point ID and index `-1`. Validation rejects missing or inconsistent source selectors.
