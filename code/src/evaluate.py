import math
import numpy as np
from surprise import accuracy


# Method for evaluation
class Evaluation:
    @staticmethod
    def calculate_rmse(predictions):
        return accuracy.rmse(predictions, False)

    @staticmethod
    # calculate average of ndcg
    # assume k <= len(top_n_df)
    def calculate_ndcg(rel_dict, top_n_df, k):
        ndcg_sum = 0
        denom = 0

        for u in rel_dict.keys():
            if u not in top_n_df.index:
                continue
            gt = rel_dict[u]
            if len(gt) == 0:
                continue

            pred = top_n_df.loc[u].tolist()

            ndcg_sum = ndcg_sum + Evaluation.cal_ndcg(gt, pred[:k])
            denom = denom + 1

        if denom == 0:  # no answer
            return math.nan
        return ndcg_sum / denom

    @staticmethod
    # calculate ndcg
    # len(prediction) = k
    def cal_ndcg(gt, prediction):
        # initialize
        dcg = 0
        k = len(prediction)

        # calculate dcg
        for idx, item in enumerate(prediction, start=1):
            if item in gt:
                dcg = dcg + 1 / math.log(idx + 1, 2)  # relevance is all 1

        # calculate idcg
        if len(gt) > k:
            idcg = sum((1 / np.log2(i + 1) for i in range(1, k+1)))
        else:
            idcg = sum((1 / np.log2(i + 1) for i in range(1, len(gt)+1)))
            # relevance of item not in gt is 0

        return dcg / idcg