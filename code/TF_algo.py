import numpy as np
from surprise import PredictionImpossible
from surprise import SVD


class SVDtf(SVD):
    def __init__(self):
        SVD.__init__(self)
        self.i_attr = None
        self.const = None
        self.n_err = None
        self.vio = None  # 0: satisfy ~ 1: not satisfy

    # pandas dataframe related to attribute and constraints
    def set_data(self, attr, const, n_err):
        self.i_attr = attr
        self.const = const
        self.n_err = n_err

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

    # u, i inner id
    def estimate(self, u, i):
        if self.vio[u][i] == 1.0:  # violate constraint
            return 0.0

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

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
                if self.vio[u][i] == 1.0:
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


    # check item satisfies constraint
    # 0: satisfy constraint, 1: not
    # assume there's only one constraint # TODO
    def check_constraint(self, item, const):
        ret = True

        if const['i1'] is not None:
            ret = self.include_ingr(item, const['i1'])
        elif const['i2'] is not None:
            ret = self.exclude_ingr(item, const['i2'])

        if ret:
            return 0.0
        else:
            return 1.0

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.i_attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return iid not in self.i_attr.loc[fid].ingredient_ids

    # return T/F for constraint 3
    # TODO
    def satisfy_nutr(self, fid, hid, target_nutr):
        nutr = self.i_attr.loc[fid].nutrition
        nutr_hist = self.i_attr.loc[hid].nutrition
        for i in range(0, len(nutr)):
            target = target_nutr[i] - float(nutr_hist[i])
            value = float(nutr[i])
            if value < (target * (1 - (self.n_err / 100))):
                return False
            if (target * (1 + (self.n_err / 100))) < value:
                return False
        return True
