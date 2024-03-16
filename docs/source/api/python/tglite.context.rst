.. _api-context

tglite.TContext
===============
.. currentmodule:: tglite
.. autoclass:: TContext

   .. automethod:: __init__


.. currentmodule:: tglite.TContext

Basic settings
--------------
.. autosummary::
   :toctree: ../../generated/

   train
   eval
   need_sampling

Set node embedding cache
------------------------
.. autosummary::
   :toctree: ../../generated/

   enable_embed_caching
   set_cache_limit

Set time precomputation
-----------------------
.. autosummary::
   :toctree: ../../generated/

   enable_time_precompute
   set_time_window

Query graph
-----------
.. autosummary::
   :toctree: ../../generated/

   graph
