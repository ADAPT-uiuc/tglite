import argparse
import os
import torch
import numpy as np
import tglite as tg

from jodie import JODIE
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
parser.add_argument('--dim-time', type=int, default=100, help='dimension of time features (default: 100)')
parser.add_argument('--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)')
parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
parser.add_argument('--move', action='store_true', help='move data to device')
args = parser.parse_args()
print(args)

device = support.make_device(args.gpu)
model_path = support.make_model_path('jodie', args.prefix, args.data)
model_mem_path = support.make_model_mem_path('jodie', args.prefix, args.data)
if args.seed >= 0:
    support.set_seed(args.seed)

DATA: str = args.data
DATA_PATH: str = args.data_path
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.bsize
LEARN_RATE: float = float(args.lr)
DIM_TIME: int = args.dim_time
DIM_EMBED: int = args.dim_embed


### load data

g = support.load_graph(os.path.join(DATA_PATH, f'data/{DATA}/edges.csv'))
support.load_feats(g, DATA, DATA_PATH)
dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
dim_nfeat = g.nfeat.shape[1]

g.mem = tg.Memory(g.num_nodes(), DIM_EMBED)
g.mailbox = tg.Mailbox(g.num_nodes(), 1, DIM_EMBED + dim_efeat)

g.set_compute(device)
if args.move:
    g.move_data(device)

ctx = tg.TContext(g)


### model

model = JODIE(ctx,
    dim_embed=DIM_EMBED,
    dim_node=dim_nfeat,
    dim_edge=dim_efeat,
    dim_time=DIM_TIME)
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


### training

train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

ctx = tg.TContext(g)
trainer = support.LinkPredTrainer(
    ctx, model, criterion, optimizer, neg_sampler,
    EPOCHS, BATCH_SIZE, train_end, val_end,
    model_path, model_mem_path)

trainer.train()
trainer.test()
