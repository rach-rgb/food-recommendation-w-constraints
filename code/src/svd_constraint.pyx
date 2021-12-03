cimport numpy as np
import numpy as np
from surprise.utils import get_rng
from surprise import PredictionImpossible
from surprise import SVD


# SVD algorithm w/ constraint
class CnstSVD(SVD):
    def __init__(self):
        SVD.__init__(self)
        self.i_attr = None
        self.const = None
        self.c_alp = None
        self.c_i1 = None
        self.c_i2 = None
        self.c_nl = None

        self.vio = None  # 0: satisfy ~ 1: not satisfy

    # pandas dataframe related to attribute and constraints
    def set_data(self, attr, cnst, c_alp, columns):
        self.i_attr = attr
        self.const = cnst
        self.c_alp = c_alp
        self.c_i1 = columns[0]
        self.c_i2 = columns[1]
        self.c_nl = columns[2]

    # check constraints for all (u, i) pairs
    def fit(self, train_set):
        SVD.fit(self, train_set)

        self.vio = np.zeros((self.trainset.n_users, self.trainset.n_items))
        for u in self.trainset.all_users():
            raw_u = int(self.trainset.to_raw_uid(u))
            const = self.const.loc[self.const.u == raw_u]
            if len(const) < 1:  # no constraint
                continue
            const = const.iloc[0]

            for i in train_set.all_items():
                raw_i = int(self.trainset.to_raw_iid(i))

                self.vio[u][i] = self.check_constraint(raw_i, const)

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            if self.vio[u][i] == -1.0:  # violate constraint
                return 0.0

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        if known_user and known_item:
            if self.vio[u][i] != 0.0: # real-number constraint exists
                est = est * (1 - self.c_alp) + (self.vio[u][i] - 1)

        return est

    # modify original SGD slightly
    def sgd(self, trainset):
        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():
                # if i violates constraint, ignore error
                if self.exclude_train(u, i):
                    err = 0
                else:
                    # compute current error
                    dot = 0  # <q_i, p_u>
                    for f in range(self.n_factors):
                        dot += qi[i, f] * pu[u, f]
                    err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi


    # score x to check item satisfies constraint
    # x = 0: satisfy perfectly or no real number constraint
    # x = -1: violate T/F constraint
    # 1 <= x <= 6: satisfy T/F constraint & (1 + score) for real number constraint
    def check_constraint(self, item, cnst):
        ret = 0.0

        if cnst['i1'] is not None:
            if self.include_ingr(item, cnst['i1']) is False:
                return -1.0
        if cnst['i2'] is not None:
            if self.exclude_ingr(item, cnst['i2']) is False:
                return -1.0
        if cnst['nl'] is not None:
            return self.apply_nutr(item, cnst['nl']) + 1

        return ret # shouldn't reach here

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.i_attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return iid not in self.i_attr.loc[fid].ingredient_ids

    # return scaled score for constraint 3
    # 0 <= score <= 5
    def apply_nutr(self, fid, target):
        nutr = self.i_attr.loc[fid].nutrition
        d = np.dot(nutr, target)
        if d == 0:
            return 0
        score = d / (np.linalg.norm(nutr) * np.linalg.norm(target)) * self.c_alp * 5
        return np.clip(score, 0, self.c_alp * 5)

    # exclude if constraint 1/2 is violated
    def exclude_train(self, u, i):
        if self.vio is not None and self.vio[u][i] == -1.0:
            return True
        return False

# train all samples
class CnstSVD_all(CnstSVD):
    def exclude_train(self, u, i):
        return False;

# exclude when constraint 1/2 violated or score from constraint 3 is less than self.c_alp * 5 * 0.5
class CnstSVD_hard(CnstSVD):
    def exclude_train(self, u, i):
        if self.vio is not None and (self.vio[u][i] == 0.0 or self.vio[u][i] >= self.c_alp * 5 * 0.5 + 1):
            return False
        elif self.vio is not None:
            return True
        return False

# exclude when constraint 1/2 violated and constraint 3 exists
class CnstSVD_harder(CnstSVD):
    def exclude_train(self, u, i):
        if self.vio is not None and self.vio[u][i] != 0.0:
            return True
        return False