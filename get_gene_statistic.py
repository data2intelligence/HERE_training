import pdb
import sys,os,shutil,glob
import numpy as np
import pandas as pd


project = sys.argv[1]
version = sys.argv[2]


alldf = pd.read_csv('{}/mutation_effects{}.csv'.format(project, version), low_memory=False)
alldf = alldf.rename(columns={'Unnamed: 0': 'Hugo_Symbol'})
gene_symbols = alldf[alldf.columns[0]].tolist()
valid_gene_inds = []
unknown_nan_counts = []
gain_counts = []
loss_counts = []
switch_counts = []
unknown_counts = []
for i, gene_symbol in enumerate(gene_symbols):
    rowdf = alldf.iloc[i]
    values = rowdf.values[1:].astype(np.str_)
    for j in range(len(values)):
        v = values[j].lower()
        if 'gain' in v:
            values[j] = 'gain'
        elif 'loss' in v:
            values[j] = 'loss'
        elif 'switch' in v:
            values[j] = 'switch'
        elif 'unknown' in v:
            values[j] = 'unknown'
        else:
            values[j] = 'nan'
    labels, counts = np.unique(values, return_counts=True)
    dd = {'unknown': 0, 'nan': 0, 'gain': 0, 'loss': 0, 'switch': 0}
    dd.update({l: c for l, c in zip(labels, counts)})
    if len(set(labels) - {'unknown', 'nan'}) > 0:
        valid_gene_inds.append(i)
        unknown_nan_counts.append(dd['unknown'] + dd['nan'])
        gain_counts.append(dd['gain'])
        loss_counts.append(dd['loss'])
        switch_counts.append(dd['switch'])
        unknown_counts.append(dd['unknown'])
valid_gene_inds = np.array(valid_gene_inds)
unknown_nan_counts = np.array(unknown_nan_counts)
gain_counts = np.array(gain_counts)
loss_counts = np.array(loss_counts)
switch_counts = np.array(switch_counts)
unknown_counts = np.array(unknown_counts)
inds = np.argsort(unknown_nan_counts)
unknown_nan_counts = unknown_nan_counts[inds]
gain_counts = gain_counts[inds]
loss_counts = loss_counts[inds]
switch_counts = switch_counts[inds]
unknown_counts = unknown_counts[inds]
valid_gene_inds = valid_gene_inds[inds]
valid_counts = len(alldf.columns) - 1 - np.array(unknown_nan_counts)
valid_df = alldf.iloc[valid_gene_inds].copy()
valid_df.insert(1, 'valid_mut_counts', valid_counts)
valid_df.insert(2, 'gain_counts', gain_counts)
valid_df.insert(3, 'loss_counts', loss_counts)
valid_df.insert(4, 'switch_counts', switch_counts)
valid_df.insert(5, 'unknown_counts', unknown_counts)
valid_df = valid_df.rename(columns={'Unnamed: 0': 'Hugo_Symbol'})
valid_df.to_csv('{}/mutation_effects_sorted2{}.csv'.format(project, version), index=False)








