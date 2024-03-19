import torch
import tglite as tg

from torch import nn, Tensor
from tglite.nn import TemporalAttnLayer
from tglite._stats import tt

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support


class TGN(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_node: int, dim_edge: int, dim_time: int, dim_embed: int,
                 sampler: tg.TSampler, num_layers=2, num_heads=2, dropout=0.1,
                 dedup: bool = True):
        super().__init__()
        self.ctx = ctx
        self.dim_edge = dim_edge
        self.num_layers = num_layers
        self.nfeat_map = None if dim_node == dim_embed else nn.Linear(dim_node, dim_embed)
        self.mem_cell = nn.GRUCell(2 * dim_embed + dim_edge + dim_time, dim_embed)
        self.mem_time_encode = tg.nn.TimeEncode(dim_time)
        self.attn = nn.ModuleList([
            TemporalAttnLayer(ctx,
                num_heads=num_heads,
                dim_node=dim_embed,
                dim_edge=dim_edge,
                dim_time=dim_time,
                dim_out=dim_embed,
                dropout=dropout)
            for i in range(num_layers)])
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

    def forward(self, batch: tg.TBatch) -> Tensor:
        # setup message passing
        head = batch.block(self.ctx)
        for i in range(self.num_layers):
            tail = head if i == 0 \
                else tail.next_block(include_dst=True, use_dst_times=False)
            tail = tg.op.dedup(tail) if self.dedup else tail
            tail = self.sampler.sample(tail)

        # load data / feats
        tg.op.preload(head, use_pin=True)
        if tail.num_dst() > 0:
            t_start = tt.start()
            mem = self.update_memory(tail)
            nfeat = tail.nfeat() if self.nfeat_map is None else self.nfeat_map(tail.nfeat())
            tail.dstdata['h'] = nfeat[:tail.num_dst()] + mem[:tail.num_dst()]
            tail.srcdata['h'] = nfeat[tail.num_dst():] + mem[tail.num_dst():]
            tt.t_mem_update += tt.elapsed(t_start)
            del nfeat
            del mem

        # compute embeddings
        embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        del head
        del tail

        # compute scores
        src, dst, neg = batch.split_data(embeds)
        scores = self.edge_predictor(src, dst)
        if neg is not None:
            scores = (scores, self.edge_predictor(src, neg))
        del embeds
        del src
        del dst
        del neg

        # memory messages
        t_start = tt.start()
        self.save_raw_msgs(batch)
        tt.t_post_update += tt.elapsed(t_start)

        return scores

    def update_memory(self, blk: tg.TBlock) -> Tensor:
        cdev = blk.g.compute_device()
        nodes = blk.allnodes()

        mail_ts = blk.g.mailbox.time[nodes]
        delta = mail_ts - blk.g.mem.time[nodes]
        delta = delta.squeeze().to(cdev)
        mail = tg.op.precomputed_times(self.ctx, 0, self.mem_time_encode, delta)
        mail = torch.cat([blk.mail(), mail], dim=1)

        mem = blk.mem_data()
        mem = self.mem_cell(mail, mem)
        blk.g.mem.update(nodes, mem, mail_ts)
        return mem

    def save_raw_msgs(self, batch: tg.TBatch):
        sdev = batch.g.storage_device()
        mem = batch.g.mem.data

        blk = batch.block_adj(self.ctx)
        blk = tg.op.coalesce(blk, by='latest')

        uniq = torch.from_numpy(blk.dstnodes).long().to(sdev)
        nbrs = torch.from_numpy(blk.srcnodes).long().to(sdev)
        if self.dim_edge > 0:
            eids = torch.from_numpy(blk.eid).long().to(sdev)
            mail = torch.cat([mem[uniq], mem[nbrs], batch.g.efeat[eids]], dim=1)
        else:
            mail = torch.cat([mem[uniq], mem[nbrs]], dim=1)
        mail_ts = torch.from_numpy(blk.ets).to(sdev)
        batch.g.mailbox.store(uniq, mail, mail_ts)