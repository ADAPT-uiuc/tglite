# TGLite - A Framework for Temporal GNNs

TGLite is a lightweight framework that provides core abstractions and building blocks for practitioners and researchers to implement efficient TGNN models. TGNNs, or Temporal Graph Neural Networks, learn node embeddings for graphs that dynamically change over time by jointly aggregating structural and temporal information from neighboring nodes. TGLite employs an abstraction called a _TBlock_ to represent the temporal graph dependencies when aggregating from neighbors, with explicit support for capturing temporal details like edge timestamps, as well as composable operators and optimizations. Compared to prior art, TGLite can outperform the [TGL][tgl] framework by [up to 3x](#publication) in terms of training time.

<div align="center">
  <img src="https://raw.githubusercontent.com/ADAPT-uiuc/tglite/main/docs/source/img/train.png">
  End-to-end training epoch time comparison on an Nvidia A100 GPU.
</div>

[tgl]: https://github.com/amazon-science/tgl

## Installation

See our [documentation][docs] for instructions on how to install the TGLite binaries, as well as examples and references for supported functionality. To install from source or for local development, go to the [Building from source](build_src) session, it also explains how to run [examples](exp).

[docs]: tglite.rtfd.io
[build_src]: docs/install/from_source.md
[exp]: https://github.com/ADAPT-uiuc/tglite/tree/main/examples

## Getting Started

TGLite is currently designed to be used with PyTorch as a training backend, typically with GPU devices. A TGNN model can be defined and trained in the usual way using PyTorch, with the computations constructed using a mix of PyTorch functions and operators/optimizations from TGLite. Below is a simple example (not a real network architecture, just for demonstration purposes):

```python
import torch
import tglite as tg

class TGNN(torch.nn.Module):
    def __init__(self, ctx: tg.TContext, dim_node=100, dim_time=100):
        super().__init__()
        self.ctx = ctx
        self.linear = torch.nn.Linear(dim_node + dim_time, dim_node)
        self.sampler = tg.TSampler(num_nbrs=10, strategy='recent')
        self.encoder = tg.nn.TimeEncode(dim_time)

    def forward(self, batch: tg.TBatch):
        blk = batch.block(self.ctx)
        blk = tg.op.dedup(blk)
        blk = self.sampler.sample(blk)
        blk.srcdata['h'] = blk.srcfeat()
        return tg.op.aggregate(blk, self.compute, key='h')

    def compute(self, blk: tg.TBlock):
        feats = self.encoder(blk.time_deltas())
        feats = torch.cat([blk.srcdata['h'], feats], dim=1)
        embeds = self.linear(feats)
        embeds = tg.op.edge_reduce(blk, embeds, op='sum')
        return torch.relu(embeds)

graph = tg.from_csv(...)
ctx = tg.TContext(graph)
model = TGNN(ctx)
train(model)
```

The example model is defined to first construct the graph dependencies for nodes in the current batch of edges. The `dedup()` optimization is applied before sampling for 10 recent neighbors. Node embeddings are computed by simply combining node and time features, applying a linear layer and summing across neighbors. More complex computations and aggregations, such as temporal self-attention often used with TGNNs, can be defined using the provided building blocks.

## Publication

* Yufeng Wang and Charith Mendis. 2024. [TGLite: A Lightweight Programming Framework for Continuous-Time Temporal Graph Neural Networks][tglite-paper]. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24), April 2024, La Jolla, CA, USA. (To Appear)

* Yufeng Wang and Charith Mendis. 2023. [TGOpt: Redundancy-Aware Optimizations for Temporal Graph Attention Networks][tgopt-paper]. In Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '23), February 2023, Montreal, QC, Canada.

If you find TGLite useful, please consider attributing to the following citation:

```bibtex
@inproceedings{wang2024tglite,
  author = {Wang, Yufeng and Mendis, Charith},
  title = {TGLite: A Lightweight Programming Framework for Continuous-Time Temporal Graph Neural Networks},
  year = {2024},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  doi = {10.1145/3620665.3640414}
}
```

[tglite-paper]: https://charithmendis.com/assets/pdf/asplos24-tglite.pdf
[tgopt-paper]: https://charithmendis.com/assets/pdf/ppopp23-tgopt.pdf
