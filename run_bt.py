import os
import pandas as pd
import numpy as np
from src.bradley_terry import ZermeloBradleyTerry, SpokoinyBradleyTerry, BayesianBradleyTerry

folder = "<path to folder with csv files>"
df = pd.DataFrame()
for file in os.listdir(folder):
    df_new = pd.read_csv(folder + file)
    df = pd.concat((df, df_new))

metrics = ['test/ndcg@20', 'test/ndcg@10', 'test/ndcg@5', 'test/recall@20',
           'test/recall@10', 'test/recall@5', 'test/coverage@20',
           'test/coverage@10', 'test/coverage@5']
algs = df['model'].unique()
tables = {}
n = len(algs)
for metric in metrics:
    table = np.zeros((n, n))
    for dataset in df['dataset'].unique():
        df_cur = df[df['dataset'] == dataset]
        for i in range(n):
            for j in range(i + 1, n):
                alg1 = df_cur[df_cur['model'] == algs[i]]
                alg2 = df_cur[df_cur['model'] == algs[j]]
                if len(alg1[metric]) == 0 or len(alg2[metric]) == 0:
                    continue
                if alg1[metric].values > alg2[metric].values:
                    table[i, j] += 1
                else:
                    table[j, i] += 1
    tables[metric] = table

print('Zermelo BT')
for metric in tables.keys():
    results = ZermeloBradleyTerry().fit(tables[metric])
    print('{:<16} {} {}'.format(metric, str(results['ranking']),
                                str(algs[results['ranking']])))
print('Bayesian BT')
for metric in tables.keys():
    W = tables[metric]
    N = W + W.T
    results = BayesianBradleyTerry(n_players=W.shape[0]).fit(W, N)
    print('{:<16} {} {}'.format(metric, str(results['ranking']),
                                str(algs[results['ranking']])))
print('Spokoiny BT')
for metric in tables.keys():
    W = tables[metric]
    N = W + W.T
    G = np.ones(W.shape)
    G[range(W.shape[0]), range(W.shape[0])] = 0
    W = W * G
    results = SpokoinyBradleyTerry(W, N, G).fit()
    print('{:<16} {} {}'.format(metric, str(results['ranking']),
                                str(algs[results['ranking']])))
