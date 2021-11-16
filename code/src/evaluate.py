import math
from surprise import accuracy


class Evaluation:
    @staticmethod
    def calculate_rmse(predictions):
        return accuracy.rmse(predictions, False)

    @staticmethod
    # assume k < top_n of RS
    def calculate_ndcg(rel_dict, top_n_df, k):
        ndcg_sum = 0
        denom = 0

        for u in rel_dict.keys():
            if u not in top_n_df.index:
                continue
            gt = rel_dict[u]
            if len(gt) > k:
                gt = gt[:k]
            prediction = top_n_df.loc[u, :(k-1)].tolist()
            ndcg_sum = ndcg_sum + Evaluation.cal_ndcg(gt, prediction)
            denom = denom + 1

        if denom == 0:  # no answer
            return math.nan
        return ndcg_sum / denom

    @staticmethod
    def cal_ndcg(gt, prediction):
        # initialize
        dcg = 0
        idcg = 0

        # calculate dcg
        for i in range(0, len(prediction)):
            item = prediction[i]
            if item in gt:
                dcg = dcg + 1 / math.log(i + 2, 2)  # relevance is all 1
            if i < len(gt):
                idcg = idcg + 1 / math.log(i + 2, 2)

        if dcg == 0:
            return 0

        return dcg / idcg