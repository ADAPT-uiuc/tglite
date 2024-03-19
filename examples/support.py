import random
import time
import os
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Callable, Optional, Tuple, Union

import tglite as tg
from tglite._stats import tt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(gpu: int) -> torch.device:
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')


def make_model_path(model: str, prefix: str, data: str) -> str:
    """If prefix is not empty, return 'models/{model}/{prefix}-{data}.pt', else return
    'models/{model}/{data}-{time.time()}.pt'."""
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}.pt'
    else:
        return f'models/{model}/{data}-{time.time()}.pt'


def make_model_mem_path(model: str, prefix: str, data: str) -> str:
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}-mem.pt'
    else:
        return f'models/{model}/{data}-mem-{time.time()}.pt'


def load_graph(path: Union[str, Path]) -> tg.TGraph:
    """Create a TGraph with edges and timestamps loaded from path. Provided data should include
    'src' 'dst' and 'time' columns."""
    df = pd.read_csv(str(path))

    src = df['src'].to_numpy().astype(np.int32).reshape(-1, 1)
    dst = df['dst'].to_numpy().astype(np.int32).reshape(-1, 1)
    etime = df['time'].to_numpy().astype(np.float32)
    del df

    edges = np.concatenate([src, dst], axis=1)
    del src
    del dst

    g = tg.TGraph(edges, etime)
    print('num edges:', g.num_edges())
    print('num nodes:', g.num_nodes())
    return g


def load_feats(g: tg.TGraph, d: str, data_path: str=''):
    """
    Load edge features and node features to g from data/{d}/edge_features.pt and
    data/{d}/edge_features.pt. If no file, create random edge and node features for data 'mooc',
    'lastfm' and 'wiki-talk', create random edge features for data 'wiki' and 'reddit', None for
    other data.
    """
    edge_feats = None
    node_feats = None

    if Path(os.path.join(data_path, f'data/{d}/edge_features.pt')).exists():
        edge_feats = torch.load(os.path.join(data_path, f'data/{d}/edge_features.pt'))
        edge_feats = edge_feats.type(torch.float32)
    elif d in ['mooc', 'lastfm', 'wiki-talk']:
        edge_feats = torch.randn(g.num_edges(), 128, dtype=torch.float32)

    if Path(os.path.join(data_path, f'data/{d}/node_features.pt')).exists():
        node_feats = torch.load(os.path.join(data_path, f'data/{d}/node_features.pt'))
        node_feats = node_feats.type(torch.float32)
    elif d in ['wiki', 'mooc', 'reddit', 'lastfm', 'wiki-talk']:
        node_feats = torch.randn(g.num_nodes(), edge_feats.shape[1], dtype=torch.float32)

    print('edge feat:', None if edge_feats is None else edge_feats.shape)
    print('node feat:', None if node_feats is None else node_feats.shape)
    g.efeat = edge_feats
    g.nfeat = node_feats


def data_split(num_samples: int, train_percent: float, val_percent: float) -> Tuple[int, int]:
    train_end = int(np.ceil(num_samples * train_percent))
    val_end = int(np.ceil(num_samples * (train_percent + val_percent)))
    return train_end, val_end


class EdgePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        h_src = self.src_fc(src)
        h_dst = self.dst_fc(dst)
        h_out = self.act(h_src + h_dst)
        return self.out_fc(h_out)


class LinkPredTrainer(object):
    def __init__(self, ctx: tg.TContext, model: nn.Module,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 neg_sampler: Callable, epochs: int, bsize: int,
                 train_end: int, val_end: int,
                 model_path: str, model_mem_path: Optional[str]):
        self.ctx = ctx
        self.g = ctx.graph
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.neg_sampler = neg_sampler
        self.epochs = epochs
        self.bsize = bsize
        self.train_end = train_end
        self.val_end = val_end
        self.model_path = model_path
        self.model_mem_path = model_mem_path

    def train(self):
        tt.csv_open('out-stats.csv')
        tt.csv_write_header()
        best_epoch = 0
        best_ap = 0
        for e in range(self.epochs):
            print(f'epoch {e}:')
            torch.cuda.synchronize()
            t_epoch = tt.start()

            self.ctx.train()
            self.model.train()
            if self.g.mem is not None:
                self.g.mem.reset()
            if self.g.mailbox is not None:
                self.g.mailbox.reset()

            epoch_loss = 0.0
            t_loop = tt.start()
            for batch in tg.iter_edges(self.g, size=self.bsize, end=self.train_end):
                t_start = tt.start()
                batch.neg_nodes = self.neg_sampler(len(batch))
                tt.t_prep_batch += tt.elapsed(t_start)

                t_start = tt.start()
                self.optimizer.zero_grad()
                pred_pos, pred_neg = self.model(batch)
                tt.t_forward += tt.elapsed(t_start)

                t_start = tt.start()
                loss = self.criterion(pred_pos, torch.ones_like(pred_pos))
                loss += self.criterion(pred_neg, torch.zeros_like(pred_neg))
                epoch_loss += float(loss)
                loss.backward()
                self.optimizer.step()
                tt.t_backward += tt.elapsed(t_start)
            tt.t_loop = tt.elapsed(t_loop)

            t_eval = tt.start()
            ap, auc = self.eval(start_idx=self.train_end, end_idx=self.val_end)
            tt.t_eval = tt.elapsed(t_eval)

            torch.cuda.synchronize()
            tt.t_epoch = tt.elapsed(t_epoch)
            if e == 0 or ap > best_ap:
                best_epoch = e
                best_ap = ap
                torch.save(self.model.state_dict(), self.model_path)
                if self.g.mem is not None:
                    torch.save(self.g.mem.backup(), self.model_mem_path)
            print('  loss:{:.4f} val ap:{:.4f} val auc:{:.4f}'.format(epoch_loss, ap, auc))
            tt.csv_write_line(epoch=e)
            tt.print_epoch()
            tt.reset_epoch()
        tt.csv_close()
        print('best model at epoch {}'.format(best_epoch))

    @torch.no_grad()
    def eval(self, start_idx: int, end_idx: int = None):
        self.ctx.eval()
        self.model.eval()
        val_aps = []
        val_auc = []
        for batch in tg.iter_edges(self.g, size=self.bsize, start=start_idx, end=end_idx):
            size = len(batch)
            batch.neg_nodes = self.neg_sampler(size)
            prob_pos, prob_neg = self.model(batch)
            prob_pos = prob_pos.cpu()
            prob_neg = prob_neg.cpu()
            pred_score = torch.cat([prob_pos, prob_neg], dim=0).sigmoid()
            true_label = torch.cat([torch.ones(size), torch.zeros(size)])
            val_aps.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
        return np.mean(val_aps), np.mean(val_auc)

    def test(self):
        print('loading saved checkpoint and testing model...')
        self.model.load_state_dict(torch.load(self.model_path))
        if self.g.mem is not None:
            self.g.mem.restore(torch.load(self.model_mem_path))
        t_test = tt.start()
        ap, auc = self.eval(start_idx=self.val_end)
        t_test = tt.elapsed(t_test)
        print('  test time:{:.2f}s AP:{:.4f} AUC:{:.4f}'.format(t_test, ap, auc))
