import math
import numpy as np
import torch
import tglite as tg

from typing import Tuple
from torch import nn, Tensor
from tglite._stats import tt

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support


class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class JODIE(nn.Module):
    def __init__(self, ctx: tg.TContext, dim_embed: int, dim_node: int, dim_edge: int, dim_time: int):
        super().__init__()
        self.ctx = ctx
        self.dim_embed = dim_embed
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time

        dim_input = dim_embed + dim_edge + dim_time
        self.updater = nn.RNNCell(dim_input, dim_embed)
        self.time_encode = tg.nn.TimeEncode(dim_time)
        self.time_linear = NormalLinear(1, dim_embed)

        if dim_node != dim_embed:
            self.node_linear = nn.Linear(dim_node, dim_embed)
        self.norm = nn.LayerNorm(dim_embed)
        self.edge_predictor = support.EdgePredictor(dim_embed)

    def forward(self, batch: tg.TBatch):
        size = len(batch)
        nodes = batch.nodes()

        t_start = tt.start()
        embed, embed_ts = self.update_embed(batch, nodes)
        tt.t_mem_update += tt.elapsed(t_start)
        embed = self.normalize_embed(batch, nodes, embed)
        batch.g.mem.update(batch.nodes(include_negs=False), embed[:2 * size], embed_ts[:2 * size])

        embed = self.project_embed(batch, embed, embed_ts)
        scores = self.edge_predictor(embed[:size], embed[size:2 * size])
        if batch.neg_nodes is not None:
            scores = (scores, self.edge_predictor(embed[:size], embed[2 * size]))
        del embed_ts
        del embed

        self.save_raw_msgs(batch)
        return scores

    def update_embed(self, batch: tg.TBatch, nodes: np.ndarray) -> Tuple[Tensor, Tensor]:
        cdev = batch.g.compute_device()

        embed_time = batch.g.mem.time[nodes]
        mail_ts = batch.g.mailbox.time[nodes]
        time_feat = (mail_ts - embed_time).to(cdev)
        time_feat = self.time_encode(time_feat.squeeze())
        input = batch.g.mailbox.mail[nodes].to(cdev)
        input = torch.cat([input, time_feat], dim=1)

        embed = batch.g.mem.data[nodes].to(cdev)
        embed = self.updater(input, embed)

        return embed, mail_ts.to(cdev)

    def normalize_embed(self, batch: tg.TBatch, nodes: np.ndarray, embed: Tensor) -> Tensor:
        nfeat = batch.g.nfeat[nodes].to(embed.device)
        if self.dim_node != self.dim_embed:
            embed = embed + self.node_linear(nfeat)
        else:
            embed = embed + nfeat
        return self.norm(embed)

    def project_embed(self, batch: tg.TBatch, embed: Tensor, embed_ts: Tensor) -> Tensor:
        times = torch.from_numpy(batch.times()).to(embed_ts.device)
        delta = times - embed_ts
        time_diff = (delta / (times + 1)).to(embed.device)
        return embed * (1 + self.time_linear(time_diff.reshape(-1, 1)))

    def save_raw_msgs(self, batch: tg.TBatch):
        sdev = batch.g.storage_device()
        blk = batch.block_adj(self.ctx)

        adj_nodes = torch.from_numpy(blk.srcnodes).long().to(sdev)
        mail = batch.g.mem.data[adj_nodes]
        if self.dim_edge > 0:
            eids = torch.from_numpy(blk.eid).long().to(sdev)
            mail = torch.cat([mail, batch.g.efeat[eids]], dim=1)
        mail_ts = torch.from_numpy(blk.ets).to(sdev)
        batch.g.mailbox.store(blk.dstnodes, mail, mail_ts)