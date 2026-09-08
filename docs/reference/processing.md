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
   :members: index, index_many, replace, results_writer, write_results, write_many_xml
```
