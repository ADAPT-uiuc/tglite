import argparse
import os
import numpy as np
import torch
import tglite as tg

from apan import APAN
import support

### arguments

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
parser.add_argument('--data-path', type=str, default='', help='path to data folder')
parser.add_argument('--prefix', type=str, default='', help='name for saving trained model')
parser.add_argument('--gpu', type=int, default=0, help='gpu device to use (or -1 for cpu)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 100)')
parser.add_argument('--bsize', type=int, default=200, help='batch size (default: 200)')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate (default: 1e-4)')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate (default: 0.1)')
# parser.add_argument('--n-layers', type=int, default=2, help='number of layers (default: 2)')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads (default: 2)')
parser.add_argument('--n-nbrs', type=int, default=20, help='number of neighbors to sample (default: 20)')
parser.add_argument('--n-mail', type=int, default=10, help='max number of mails (default: 10)')
parser.add_argument('--dim-time', type=int, default=100, help='dimension of time features (default: 100)')
parser.add_argument('--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)')
parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
parser.add_argument('--move', action='store_true', help='move data to device')
parser.add_argument('--n-threads', type=int, default=32, help='number of threads for sampler (default: 32)')
parser.add_argument('--sampling', type=str, default='recent', choices=['recent', 'uniform'], help='sampling strategy (default: recent)')
parser.add_argument('--opt-time', action='store_true', help='enable precomputing time encodings')
parser.add_argument('--time-window', type=str, default=1e4, help='time window to precompute (default: 1e4)')
parser.add_argument('--opt-all', action='store_true', help='enable all available optimizations')
args = parser.parse_args()
print(args)

device = support.make_device(args.gpu)
model_path = support.make_model_path('apan', args.prefix, args.data)
model_mem_path = support.make_model_mem_path('apan', args.prefix, args.data)
if args.seed >= 0:
    support.set_seed(args.seed)

DATA: str = args.data
DATA_PATH: str = args.data_path
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.bsize
LEARN_RATE: float = float(args.lr)
DROPOUT: float = float(args.dropout)
# N_LAYERS: int = args.n_layers
N_HEADS: int = args.n_heads
N_NBRS: int = args.n_nbrs
N_MAIL: int = args.n_mail
DIM_TIME: int = args.dim_time
DIM_EMBED: int = args.dim_embed
N_THREADS: int = args.n_threads
SAMPLING: str = args.sampling
OPT_TIME: bool = args.opt_time or args.opt_all
TIME_WINDOW: int = int(args.time_window)


### load data

g = support.load_graph(os.path.join(DATA_PATH, f'data/{DATA}/edges.csv'))
support.load_feats(g, DATA, DATA_PATH)
dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
g.nfeat = None
dim_mem = DIM_EMBED

g.mailbox = tg.Mailbox(g.num_nodes(), N_MAIL, 2 * dim_mem + dim_efeat)
g.mem = tg.Memory(g.num_nodes(), dim_mem)

g.set_compute(device)
if args.move:
    g.move_data(device)

ctx = tg.TContext(g)
ctx.need_sampling(True)
ctx.enable_time_precompute(OPT_TIME)
ctx.set_time_window(TIME_WINDOW)


### model

sampler = tg.TSampler(N_NBRS, strategy=SAMPLING, num_threads=N_THREADS)
model = APAN(ctx,
    dim_mem=dim_mem,
    dim_edge=dim_efeat,
    dim_time=DIM_TIME,
    sampler = sampler,
    num_heads=N_HEADS,
    dropout=DROPOUT)
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


### training

train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

trainer = support.LinkPredTrainer(
    ctx, model, criterion, optimizer, neg_sampler,
    EPOCHS, BATCH_SIZE, train_end, val_end,
    model_path, model_mem_path)

trainer.train()
trainer.test()