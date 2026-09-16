# Processing

The preferred API either indexes one frame with `index_frame` or reuses an
`Indexer` across frames that share configuration.

## One-Off Indexing

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autofunction:: index_frame
```

## Reusable Indexer

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: Indexer
   :members: index, index_many, iter_index, replace, results_writer, write_results, write_many_xml
```

## Parallel Indexing

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: FrameInput

.. autoclass:: FrameOutcome
   :members: ok

.. autoclass:: FrameOutcomes
   :members: close

.. autodata:: EXPECTED_INPUT_ERRORS
   :no-value:
```

`EXPECTED_INPUT_ERRORS` are the exception types that describe one input and are returned in `FrameOutcome.error` rather than raised; see [Batch indexing](../guides/batch-indexing.md).

## Frame Inputs

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autofunction:: load_mask

.. autodata:: lauelab.indexing.indexer.SUPPORTED_FRAME_DTYPES
   :no-value:
```

`SUPPORTED_FRAME_DTYPES` lists the NumPy dtypes that `Indexer.index` accepts for a frame; see [Frame input](../guides/frame-input.md).
