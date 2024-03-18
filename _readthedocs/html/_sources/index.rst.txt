.. TGLite documentation master file, created by
   sphinx-quickstart on Wed Nov  1 15:04:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TGLite's documentation!
==================================

**TGlite** is a lightweight framework that provides core abstractions and building blocks for practitioners and researchers to implement efficient TGNN models. TGNNs, or Temporal Graph Neural Networks, learn node embeddings for graphs that dynamically change over time by jointly aggregating structural and temporal information from neighboring nodes. 

TGLite employs an abstraction called a TBlock to represent the temporal graph dependencies when aggregating from neighbors, with explicit support for capturing temporal details like edge timestamps, as well as composable operators and optimizations. Compared to prior art, TGLite can outperform the `TGL <https://github.com/amazon-science/tgl>`_ framework by up to 3x in terms of training time.


.. toctree::
   :maxdepth: 1
   :caption: TGLite
   :hidden:
   :glob:

   install/index
   tutorial/TGAT

.. toctree::
   :maxdepth: 2
   :caption: API
   :glob:

   api/python/tglite
   api/python/tglite.batch
   api/python/tglite.block
   api/python/tglite.context
   api/python/tglite.graph
   api/python/tglite.sampler
   api/python/tglite.mailbox
   api/python/tglite.memory
   api/python/tglite.nn


.. note::
   This project is under active development.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
