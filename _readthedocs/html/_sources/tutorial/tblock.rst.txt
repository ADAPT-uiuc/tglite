.. _tblock_tutorial:

How does TBlock work?
=====================

Introduction
------------

This tutorial provides an overview of `TBlock`, a key component of the TGLite framework. TBlocks, or temporal blocks, capture the message-flow dependencies between target node-time pairs and their temporally sampled neighbors. This tutorial will explain the design choices and features of TBlock to help you understand its usage within the TGLite framework.

Overview
--------

TBlocks are motivated by the MFG (Message Flow Graph) objects available in the DGL (Deep Graph Library) but provide additional capabilities for CTDG (Continuous-Time Dynamic Graph) models. The following sections will explain the three key design choices that distinguish TBlocks from MFGs.

Doubly-Linked List Structure
----------------------------

One key distinction of TBlocks is the use of a doubly-linked list structure. This structure explicitly captures the multi-hop neighbor sampling/aggregation relationship that TBlocks are used for. Unlike standalone MFG objects in DGL/TGL, TBlocks maintain links to related blocks, enabling efficient multi-hop aggregation operations.

Target and Neighbor Information
-------------------------------

TBlocks primarily focus on target destination nodes and optionally include information about neighbor source nodes. By separating neighbor information as optional, TBlocks allow for easier manipulation of target node information. This flexibility enables optimizations such as deduplication and caching to be applied effectively, as they can be performed on destination nodes before sampling for neighbors.

Hooks Mechanism for Post-Processing
-----------------------------------

TBlocks provide a hooks mechanism for running post-processing procedures. These hooks are callable functions that are invoked after computations are performed on the block. The hooks mechanism enables scheduling of transformations on computed output, such as deduplication and preserving output semantics. TGLite runtime automatically handles the execution of registered hooks, simplifying the post-processing step.

.. _tblock-structure figure:
.. figure:: ../img/tblock-structure.png
   :alt: tblock-structure
   :align: center
   :figwidth: 60 %

   Diagram of the doubly-linked list design and internal structure of a TBlock (destination node-time is denoted as <i,t>).

Block Lifecycle and Usage
-------------------------

The block lifecycle involves creating a TBlock, applying optimizations, sampling neighbors, performing computations, and accessing cached data. The following steps outline the typical usage of a TBlock:

1. Create a TBlock using various methods or construct it directly.
2. Apply optimizations to the block to minimize subgraph size and potential computations.
3. Sample neighbors of the block to capture message-flow dependencies.
4. Manipulate the block in-place, register hooks, or cache data.
5. Use the block for computations and access cached data for specific nodes and edges.

.. _tblock-workflow figure:
.. figure:: ../img/tblock-workflow.png
   :alt: tblock-workflow
   :align: center
   :figwidth: 60 %

   Typical flow of constructing and using a TBlock object.

Example Usage
-------------

Here's an example code snippet demonstrating the usage of TBlock:

.. code-block:: python
   
   # Create the head TBlock from a batch data
   head = batch.block(self.ctx)

   # Create the next TBlock iteratively
   for i in range(self.num_layers):
      tail = head if i == 0 else tail.next_block(...)
      # Apply optimizations
      tail = tg.op.dedup(tail)
      tail = tg.op.cache(self.ctx, tail, ...)
      # Sample neighbors
      tail = self.sampler.sample(tail)
    
   # Load data
   tg.op.preload(head, use_pin=True)
   tail.dstdata['h'] = tail.dstfeat()
   tail.srcdata['h'] = tail.srcfeat()
   # Perform computations
   emb = tg.op.aggregate(head, self.compute, key='h')

    

In this tutorial, you learned about TBlock, a key component of the TGLite framework. TBlocks provide a powerful mechanism for capturing and analyzing message-flow dependencies in a continuous-time dynamic graph. By understanding the design choices and features of TBlock, you can effectively leverage its capabilities within your applications.

For more details and advanced usage, refer to the :ref: `TBlock <api-block>`.