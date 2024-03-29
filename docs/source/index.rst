.. TGLite documentation master file, created by
   sphinx-quickstart on Wed Nov  1 15:04:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TGLite's documentation!
==================================

**TGLite** is a lightweight framework that provides core abstractions and building blocks for practitioners and researchers to implement efficient TGNN models. *TGNNs*, or *Temporal Graph Neural Networks*, learn node embeddings for graphs that dynamically change over time by jointly aggregating structural and temporal information from neighboring nodes. 

TGLite employs an abstraction called a :ref:`TBlock <api-block>` to represent the temporal graph dependencies when aggregating from neighbors, with explicit support for capturing temporal details like edge timestamps, as well as composable operators and optimizations. Compared to prior art, TGLite can outperform the `TGL <https://github.com/amazon-science/tgl>`_ framework by up to *3x* in terms of training time.

.. _train figure:
.. figure:: img/train.png
   :alt: End-to-end training epoch time comparison on an Nvidia A100 GPU
   :align: center
   :figwidth: 85 %

   End-to-end training epoch time comparison on an Nvidia A100 GPU

Install TGLite
--------------
See :ref:`Getting started <getting-started>` for instructions on how to install the TGLite binaries. To install from source or for local development, refer to :ref:`Building from source <build-from-source>` and :ref:`Development mode <development-mode>`.

Tutorials
---------
We provide a set of tutorials to help you get started with TGLite. These tutorials cover the basics of using TGLite, as well as more advanced topics.

0. Quickstart_: A step-by-step guide to train a TGNN model using TGLite.
1. :ref:`How does TBlock work? <tutorial-tblock>`: A tutorial on how to use the TBlock abstraction to implement TGNN models.

.. _Quickstart: tutorial/quickstart.ipynb

.. toctree::
   :maxdepth: 1
   :caption: TGLite
   :hidden:
   :glob:

   install/index
   tutorial/quickstart
   tutorial/tblock

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
   api/python/tglite.op


.. note::
   This project is under active development.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
