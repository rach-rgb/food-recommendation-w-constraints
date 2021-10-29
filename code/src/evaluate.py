import pandas as pd
import math


class NDCG():
    def __init__(self, answ_length=10, res_length=10):
        self.alen = answ_length
        self.rlen = res_length
        self.rel = [3 for i in range (0, int(self.alen / 3))]
        self.rel = self.rel + [2 for i in range (0, int(self.alen / 3))]
        self.rel = self.rel + [1 for i in range (0, self.alen - int(self.alen / 3) * 2)]
        self.idcg = self.cal_idcg(self.rlen)

    # calculate IDCG
    def cal_idcg(self, p):
        assert(p <= self.alen)

        idcg = 0
        # calculate IDCG
        for i in range (0, p):
            idcg = idcg + self.rel[i] / math.log(i + 2, 2)

        return idcg

    # calculate NDCG between answer and result
    # assume len(result) < rel_length
    def cal_ndcg(self, answer, result):
        # initialize
        dcg = 0

        # calculate dcg
        for item in result:
            if item in answer:
                answ_idx = answer.index(item)
                res_idx = result.index(item)
                dcg = dcg + (self.rel[answ_idx] / math.log(res_idx + 2, 2))

        if dcg == 0:
            return 0

        return dcg / self.idcg

    # calculate average ndcg for dataframe adf(answer) and rdf(result)
    def avg_ndcg(self, adf, rdf):
        avg = 0

        for u in rdf.index:
            answ = adf.loc[u, :self.alen-1].tolist()
            result = rdf.loc[u, :self.rlen - 1].tolist()
            avg = avg + self.cal_ndcg(answ, result)

        return avg / len(rdf.index)
