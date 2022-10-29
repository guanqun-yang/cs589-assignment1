import math
import pathlib
import itertools

import numpy as np
import pandas as pd

from collections import defaultdict


def mrr(ranking):
    # make sure the ranking contains numeric or boolean relevance
    assert all(rel * 0 == 0 for rel in ranking),\
    "make sure input array has only numeric or boolean types"

    score, rel_cnt = 0, 0
    for idx, rel in enumerate(ranking):
        if rel == 0: continue
        
        score += 1 / (idx + 1)
        rel_cnt += 1

    if rel_cnt == 0:
        return 0
    else:
        return score / rel_cnt


def ndcg(ranking, topK=4):
    # make sure the ranking contains numeric or boolean relevance
    assert all(rel * 0 == 0 for rel in ranking),\
    "make sure input array has only numeric or boolean types"

    # compute dcg
    dcg = 0
    for idx, rel in enumerate(ranking[:topK]):
        rank = idx + 1
        dcg += (2 ** rel - 1) / np.log2(rank + 1)
    
    # compute idcg
    idcg, max_rel = 0, max(ranking)
    rel_cnt = sum([math.isclose(rel, 0) == False for rel in ranking])
    for idx in range(min(rel_cnt, topK)):
        rank = idx + 1
        idcg += (2 ** max_rel - 1) / np.log2(rank + 1)
    
    score = dcg / idcg
    return score


def load_retrieval_result(lang, algo, component):
    filename = f"result/{lang}_{algo}_{component}.txt"
    df = pd.read_csv(filename, sep="\t", dtype={"qid1": str, "qid2": str, "score": float, "label": int})

    return df


def compute_final_metric(lang, algo, component, metric="mrr"):
    assert metric in ["mrr", "ndcg@5", "ndcg@10"], "make sure the metric is one of 'mrr', 'ndcg@5, and 'ndcg@10'"   

    # if result file does not exists, return -1
    filename = f"result/{lang}_{algo}_{component}.txt"
    if not pathlib.Path(filename).exists(): return -1

    df = load_retrieval_result(lang, algo, component)

    if metric in ["ndcg@5", "ndcg@10"]:
        topK = int(metric.split("@")[-1])

    score = 0
    for qid1 in df.qid1.unique():
        ranking_lst = df[df.qid1 == qid1].label.tolist()
        score_lst = df[df.qid1 == qid1].score.tolist()

        # make adjustment to score
        score_lst = [s - r * 1e-6 for s, r in zip(score_lst, ranking_lst)]
        # rank the relvance score (i.e., label) based on descending order of score
        ranking_lst = [r for r, _ in sorted(zip(ranking_lst, score_lst), key=lambda x: x[1], reverse=True)]

        if metric == "mrr": 
            score += mrr(ranking_lst)
        else: 
            score += ndcg(ranking_lst, topK=topK)
        
    return score / df.qid1.nunique()




    