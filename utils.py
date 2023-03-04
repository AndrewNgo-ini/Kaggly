#%%writefile utils.py

import heapq
from dataclasses import dataclass

import torch
from typing import Callable
import numpy as np

def cos_sim(a, b):
    # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


# From: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L204
def semantic_search(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[
                    corpus_start_idx : corpus_start_idx + corpus_chunk_size
                ],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, len(cos_scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(
                            queries_result_list[query_id], (score, corpus_id)
                        )

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {
                "corpus_id": corpus_id,
                "score": score,
            }
        queries_result_list[query_id] = sorted(
            queries_result_list[query_id], key=lambda x: x["score"], reverse=True
        )

    return queries_result_list
   
    
def get_topk_preds(topic_embeddings, content_embeddings, df, k=100, return_idxs=True):
    """
    df has the same number of rows as topic_embeddings and content_embeddings.
    A bunch of the topic embeddings are duplicates, so we'll dedupe before finding 
    nearest neighbors.
    
    Returns tuple (prediction content ids, topic ids)
    
        prediction content ids is a list of lists. The outer list has the same number of elements as unique topic
        ids. The inner list contains content ids and the length is equal to k (num nearest neighbors).
    """
    
    # These idx values will be used to compare with the idx
    # values returned by `semantic_search` to calculate recall.
    df["idx"] = list(range(len(content_embeddings)))
    
    deduped_topics = df[["idx", "topic_id"]].drop_duplicates("topic_id")
    topic_embeddings = topic_embeddings[deduped_topics["idx"]]
    device = torch.device("cuda:0")
    
    deduped_content = df[["idx", "content_id"]].drop_duplicates("content_id")
    content_ids = deduped_content.content_id.values
    content_embeddings = content_embeddings[deduped_content["idx"]]
    

    # Compare each of the topic embeddings to each of the
    # content embeddings and return a ranking for each one.
    # Works much, much faster on GPU.
    search_results = semantic_search(
        torch.tensor(topic_embeddings, device=device),
        torch.tensor(content_embeddings, device=device),
        top_k=k,
    )
    
    # `search_results` is a list of lists. The inner list
    # has a `dict` at each element.
    # The dict has two keys: `corpus_id` and `score`.
    all_pred_c_ids = [[content_ids[x["corpus_id"]] for x in row] for row in search_results]
    
    return all_pred_c_ids, deduped_topics["topic_id"].tolist()

def precision_score(pred_content, gt_content):
    """
    Arguments can be int (idx) or string values.
    """
    def precision(pred, gt):
        tp = len(set(pred)&set(gt))
        fp = len(set(pred)-set(gt))
        return tp/(tp+fp+1e-7)
    
    # Get a recall score for each row of the dataset
    return [
        precision(pred, gt)
        for pred, gt in zip(pred_content, gt_content)
    ]

def recall_score(pred_content, gt_content):
    """
    Arguments can be int (idx) or string values.
    """
    def recall(pred, gt):
        tp = len(set(pred)&set(gt))
        return tp/len(set(gt))
    
    # Get a recall score for each row of the dataset
    return [
        recall(pred, gt)
        for pred, gt in zip(pred_content, gt_content)
    ]

def mean_f2_score(precision_scores, recall_scores):
    """
    Inputs should be outputs of the `precision_score` and 
    `recall_score` functions.
    """
    beta = 2
    
    def f2_score(precision, recall):
        return (1+beta**2)*(precision*recall)/(beta**2*precision+recall+1e-7)
    
    return round(np.mean([f2_score(p, r) for p, r in zip(precision_scores, recall_scores)]), 5)
    
    

def compute_metrics(eval_predictions, val_df, k=100, filter_by_lang=True):
    """
    After creating embeddings for all of the topic and content texts,
    perform a semantic search and measure the recall@100. The model
    has not seen any of these examples before, so it should be a
    good measure of how well the model can generalize.

    Since the dataset uses the exploded view of the correlations
    (one topic with 5 contents is 5 rows), I need to deduplicate
    the topic embeddings. Then I can use the `semantic_search`
    function taken from the sentence-transformers util to
    do a cosine similarity search of the topic embeddings with all
    content embeddings. This function conveniently returns the top
    `k` indexes, which makes it easy to compare with the true indexes.
    """
    
    if isinstance(k, int):
        k = [k]

    # eval_predictions is a tuple of (model_output, labels)
    # The model_output is whatever is returned by `compute_loss`
    (topic_embeddings, content_embeddings), _ = eval_predictions

    pred_content_ids, topic_ids = get_topk_preds(topic_embeddings, content_embeddings, val_df, k=max(k))
    
    # Filter based on language
    
    if filter_by_lang:
        content2lang = {content_id: lang for content_id, lang in val_df[["content_id", "content_language"]].values}
        topic2lang = {topic_id: lang for topic_id, lang in val_df[["topic_id", "topic_language"]].values}

        filtered_pred_content_ids = []
        for preds, topic_id in zip(pred_content_ids, topic_ids):
            topic_lang = topic2lang[topic_id]
            filtered_pred_content_ids.append([c_id for c_id in preds if content2lang[c_id]==topic_lang])
            
        pred_content_ids = filtered_pred_content_ids
    
    
    # Make sure true content ids are in same order as predictions
    grouped = val_df[["topic_id", "content_id"]].groupby("topic_id").agg(list)
    true_content_ids = grouped.loc[topic_ids].reset_index()["content_id"]
    
    metrics = {}
    for kk in k:
        top_preds = [row[:kk] for row in pred_content_ids]
        precisions = precision_score(top_preds, true_content_ids)
        recalls = recall_score(top_preds, true_content_ids)
        f2 = mean_f2_score(precisions, recalls)
        
        metrics[f"recall@{kk}"] = np.mean(recalls)
        metrics[f"f2@{kk}"] = f2
    
    return metrics