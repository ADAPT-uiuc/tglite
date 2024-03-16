import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support


class TGAT(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_node: int, dim_edge: int, dim_time: int, dim_embed: int,
                 sampler: tg.TSampler, num_layers=2, num_heads=2, dropout=0.1,
                 dedup: bool = True):
        super().__init__()
        self.ctx = ctx
        self.num_layers = num_layers
        self.attn = nn.ModuleList([
            TemporalAttnLayer(ctx,
                num_heads=num_heads,
                dim_node=dim_node if i == 0 else dim_embed,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_embed,
                dropout=dropout)
            for i in range(num_layers)])
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

    def forward(self, batch: tg.TBatch) -> Tensor:
        head = batch.block(self.ctx)
        for i in range(self.num_layers):
            tail = head if i == 0 \
                else tail.next_block(include_dst=True)
            tail = tg.op.dedup(tail) if self.dedup else tail
            tail = tg.op.cache(self.ctx, tail.layer, tail)
            tail = self.sampler.sample(tail)

        tg.op.preload(head, use_pin=True)
        if tail.num_dst() > 0:
            tail.dstdata['h'] = tail.dstfeat()
            tail.srcdata['h'] = tail.srcfeat()
        embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        del head
        del tail

        src, dst, neg = batch.split_data(embeds)
        scores = self.edge_predictor(src, dst)
        if batch.neg_nodes is not None:
            scores = (scores, self.edge_predictor(src, neg))

        return scores