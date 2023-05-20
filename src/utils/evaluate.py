import numpy as np
import math
import random




def rel_results(predictions, targets, scores, k):
    results = []
    batch_length = len(targets)
    for b in range(batch_length):
        one_batch_sequence = predictions[
            b * k : (b + 1) * k
        ]
        one_batch_score = scores[
            b * k : (b + 1) * k
        ]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        gt = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == gt:
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    return results

def get_metrics_results(rel_results, metrics):
    res = []
    for m in metrics:
        if m.lower().startswith('hit'):
            k = int(m.split('@')[1])
            res.append(hit_at_k(rel_results, k))
        elif m.lower().startswith('ndcg'):
            k = int(m.split('@')[1])
            res.append(ndcg_at_k(rel_results, k))
    
    return np.array(res)

def ndcg_at_k(relevance, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in relevance:
        rel = row[:k]
        one_ndcg = 0.0
        for i in range(len(rel)):
            one_ndcg += rel[i] / math.log(i+2,2)
        ndcg += one_ndcg
    return ndcg
        
    
def hit_at_k(relevance, k):
    correct = 0.0
    for row in relevance:
        rel = row[:k]
        if sum(rel) > 0:
            correct += 1
    return correct
        