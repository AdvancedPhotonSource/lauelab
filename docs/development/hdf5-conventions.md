# HDF5 file conventions

These conventions apply to HDF5 files whose layout `lauelab` defines. The indexing results file follows them, and a future single-file reconstruction output must too, so that one reader policy covers both. Detector frames and per-depth reconstruction outputs follow the 34-ID-E `entry1` layout, which `lauelab` reads and writes but does not define.

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

`version` is one integer. Adding a dataset or an attribute does not change it, and a reader ignores names it does not know. Removing, renaming, or reinterpreting a dataset increments it, and a reader raises for a version it does not support. Record the reason for a version change in the layout module and the documentation in the same commit.

## Units

Every dimensioned dataset carries a `units` attribute using the symbols from the documentation style: `um`, `nm`, `1/nm`, `deg`, `keV`, `pixel`, `s`. A dimensionless dataset has no `units` attribute. A value whose unit the acquisition metadata does not establish is kept with `units="unspecified"`; readers must not infer a unit or convert it. A convention a unit cannot express is a further attribute, for example that reciprocal-lattice rows are `a*`, `b*`, `c*` and include the factor of two pi.

## Ragged data

Store variable-length per-record data as one flat dataset plus an offsets dataset that starts at zero, ends at the total row count, and has one more entry than there are owners. The rows of owner `i` are `offsets[i]` to `offsets[i + 1]`. Do not use variable-length datasets or region references for numeric data; they cannot be read as one array.

## Storage

Store datasets uncompressed unless the writer offers compression as an option; compressed reads cost several times more and these files are small next to the frames they summarize. A writer that appends records uses chunked datasets with an unlimited first dimension, which costs a few milliseconds per read.

## Layout definition

Define the complete layout of a format in one module: every dataset path, dtype, shape, units, and fixed attribute. The writer, reader, converter, and reference documentation read that one table, and a test compares it against the documented layout. The indexing results layout is `lauelab/_results_layout.py`; the shared root-attribute and version helpers are `lauelab/_hdf5.py`.
