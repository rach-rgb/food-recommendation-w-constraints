import math
from surprise import accuracy


class Evaluation:
    @staticmethod
    def calculate_rmse(predictions):
        return accuracy.rmse(predictions, False)

    @staticmethod
    def calculate_ndcg(rel_dict, top_n_df, k):
        ndcg_sum = 0
        ndcg_target = 0

        for u in top_n_df.index:
            gt = rel_dict[u]
            if len(gt) > k:
                gt = gt[:k]
            if len(gt) == 0:
                continue # skip
            prediction = top_n_df.loc[u, :(k-1)].tolist()
            ndcg_sum = ndcg_sum + Evaluation.cal_ndcg(gt, prediction)
            ndcg_target = ndcg_target + 1

        return ndcg_sum / ndcg_target

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