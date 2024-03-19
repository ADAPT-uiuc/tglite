import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
args = parser.parse_args()


def gen_edges_wiki_talk():
    df = pd.read_csv('data/wiki-talk/wiki-talk-temporal.txt',
         sep=' ', header=None, names=['src', 'dst', 'time'],
         dtype={'src': np.int32, 'dst': np.int32, 'time': np.float32})

    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    num_edges = df.shape[0]
    train_end = int(np.ceil(num_edges * 0.70))
    valid_end = int(np.ceil(num_edges * 0.85))
    print('num_nodes:', num_nodes)
    print('num_edges:', num_edges)
    print('train_end:', train_end)
    print('valid_end:', valid_end)

    df['int_roll'] = np.zeros(num_edges, dtype=np.int32)
    ext_roll = np.zeros(num_edges, dtype=np.int32)
    ext_roll[train_end:] = 1
    ext_roll[valid_end:] = 2
    df['ext_roll'] = ext_roll

    df.to_csv('data/wiki-talk/edges.csv')


if args.data == 'wiki-talk':
    gen_edges_wiki_talk()
else:
    print('not handling dataset:', args.data)
