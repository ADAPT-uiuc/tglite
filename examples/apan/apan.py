import torch
import tglite as tg

from torch import nn, Tensor
from tglite._stats import tt

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..')) 
import support


class APAN(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_mem: int, dim_edge: int, dim_time: int,
                 sampler: tg.TSampler, num_heads=2, dropout=0.1):
        super().__init__()
        self.ctx = ctx
        self.dim_edge = dim_edge
        self.mem_updater = AttnMemoryUpdater(ctx,
            dim_mem=dim_mem,
            dim_msg=2 * dim_mem + dim_edge,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=dropout)
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim_mem)

    def forward(self, batch: tg.TBatch):
        size = len(batch)
        t_start = tt.start()
        mem = self.mem_updater(batch)
        tt.t_mem_update += tt.elapsed(t_start)

        nodes = batch.nodes(include_negs=False)
        times = batch.times(include_negs=False)
        batch.g.mem.update(nodes, mem[:2 * size], torch.from_numpy(times))

        src, dst, neg = batch.split_data(mem)
        scores = self.edge_predictor(src, dst)
        if batch.neg_nodes is not None:
            scores = (scores, self.edge_predictor(src, neg))
        del src
        del dst
        del neg

        blk = tg.TBlock(self.ctx, 0, nodes, times)
        blk = self.sampler.sample(blk)
        mem = mem[:2 * size].detach().to(batch.g.storage_device())
        self.create_mails(batch, blk, mem)
        del mem

        tg.op.propagate(blk, self.send_mails)
        return scores

    def create_mails(self, batch: tg.TBatch, blk: tg.TBlock, mem: Tensor):
        size = len(batch)
        mem_src = mem[:size]
        mem_dst = mem[size:]
        
        if self.dim_edge > 0:
            efeat = batch.g.efeat[batch.eids()]
            src_mail = torch.cat([mem_src, mem_dst, efeat], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, efeat], dim=1)
        else:
            src_mail = torch.cat([mem_src, mem_dst], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src], dim=1)

        blk.dstdata['mail'] = torch.cat([src_mail, dst_mail], dim=0)

    def send_mails(self, blk: tg.TBlock):
        sdev = blk.g.storage_device()
        if blk.num_edges() == 0:
            return

        mail = blk.dstdata['mail'][blk.dstindex]
        mail = tg.op.src_scatter(blk, mail, op='mean')

        mail_ts = torch.from_numpy(blk.dsttimes)
        mail_ts = mail_ts.to(sdev)[blk.dstindex]
        mail_ts = tg.op.src_scatter(blk, mail_ts, op='mean')

        blk.g.mailbox.store(blk.uniq_src()[0], mail, mail_ts)


class AttnMemoryUpdater(nn.Module):
    def __init__(self, ctx: tg.TContext,
                 dim_mem: int, dim_msg: int, dim_time: int,
                 num_heads=2, dropout=0.1):
        super().__init__()
        assert (dim_mem % num_heads == 0)
        self.ctx = ctx
        self.num_heads = num_heads
        self.time_encode = tg.nn.TimeEncode(dim_time)
        self.w_q = nn.Linear(dim_mem, dim_mem)
        self.w_k = nn.Linear(dim_msg + dim_time, dim_mem)
        self.w_v = nn.Linear(dim_msg + dim_time, dim_mem)
        self.mlp = nn.Linear(dim_mem, dim_mem)
        self.attn_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_mem)

    def forward(self, batch: tg.TBatch) -> Tensor:
        sdev = batch.g.storage_device()
        cdev = batch.g.compute_device()
        nodes = batch.nodes()
        times = batch.times()

        size = len(nodes)
        mem = batch.g.mem
        mailbox = batch.g.mailbox
        mail_size = mailbox.dims()[0]

        mem_data = mem.data[nodes].to(cdev)
        Q = self.w_q(mem_data).reshape(size, self.num_heads, -1)

        mail = mailbox.mail[nodes].to(cdev)
        mail = mail.reshape(size, mail_size, -1)
        time_feat = torch.from_numpy(times).to(sdev).reshape(-1, 1)
        time_feat = (time_feat - mailbox.time[nodes]).to(cdev)
        time_feat = tg.op.precomputed_times(self.ctx, 0, self.time_encode, time_feat)
        time_feat = time_feat.reshape(size, mail_size, -1)
        mail = torch.cat([mail, time_feat], dim=2)
        del time_feat

        K = self.w_k(mail).reshape(size, mail_size, self.num_heads, -1)
        V = self.w_v(mail).reshape(size, mail_size, self.num_heads, -1)
        del mail

        attn = torch.sum(Q[:, None, :, :] * K, dim=3)
        del Q

        attn = self.attn_act(attn)
        attn = nn.functional.softmax(attn, dim=1)
        attn = self.dropout(attn)
        out = torch.sum(attn[:, :, :, None] * V, dim=1)
        del attn

        out = out.reshape(size, -1)
        out = out + mem_data
        del mem_data

        out = self.layer_norm(out)
        out = self.mlp(out)
        out = self.dropout(out)
        out = nn.functional.relu(out)

        return out
