#!/usr/bin/env bash

wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/int_train.npz
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/int_full.npz
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/ext_full.npz
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
wget -P ./data/gdelt https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt
wget -P ./data/lastfm https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv
wget -P ./data/lastfm https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/ext_full.npz
wget -P ./data/lastfm https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_full.npz
wget -P ./data/lastfm https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_train.npz
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/int_train.npz
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/labels.csv
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/int_full.npz
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/ext_full.npz
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv
# wget -P ./data/mag https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/node_features.pt
wget -P ./data/mooc https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv
wget -P ./data/mooc https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/ext_full.npz
wget -P ./data/mooc https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_full.npz
wget -P ./data/mooc https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/int_train.npz
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/ext_full.npz
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_full.npz
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_train.npz
wget -P ./data/reddit https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/ext_full.npz
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_full.npz
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_train.npz
wget -P ./data/wiki https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv

wget -P ./data/wiki-talk http://snap.stanford.edu/data/wiki-talk-temporal.txt.gz
cd ./data/wiki-talk && gzip -d wiki-talk-temporal.txt.gz
