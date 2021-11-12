import pandas as pd
import time
import copy
from sys import path

from surprise import accuracy

# setting path
path.append('../src')

# setting path
from src import inter_rec
from src import post_rec
from src import TF_algo

# constants
# file path
input_path = "../../data/"
output_path = '../../result/'
rate_file = 'reduced_rating_data.csv'
attr_file = 'recipe_data.csv'

# user/item max
user_max = 1000
food_max = 5000

# const count
const_count = 3

def run_inter(ctype, idx, save_result=True):
    # PostRec applies constraint after the rating of each item is predicted
    rec = inter_rec.InterRec(input_path + rate_file, input_path + attr_file,
                             input_path + 'const_' + str(ctype) + '.' + str(idx) + '.csv',
                             TF_algo.SVDtf(), split=True)

    rec.get_data()  # get rating, attribute, recipe data

    # train with data
    start = time.time()
    rec.train()
    t1 = time.time() - start

    # predict rating for test-set
    predict_test = rec.test_rmse()
    r = accuracy.rmse(predict_test, False)

    # get top-n for anti-test-set
    start = time.time()
    rec.test()
    top_n_df = rec.get_top_n()
    t2 = time.time() - start

    if save_result:
        top_n_df.to_csv(output_path + 'InterRec' + str(ctype) + '.' + str(idx) + '.csv')

    return r, t1, t2


if __name__ == "__main__":
    # Dictionary keys
    rs1 = 'w/o Constraint'
    rs2 = 'Post-Single-'
    rs3 = 'Inter-Single-'
    val_r = 'RMSE'
    val_t1 = 'train time(s)'
    val_t2 = 'exec time(s)'

    val_dict = {
        val_r: 0,
        val_t1: 0,
        val_t2: 0
    }

    keys = [rs1]
    keys = keys + [rs2 + str(i) for i in range(1, 4)]
    keys = keys + [rs3 + str(i) for i in range(1, 4)]

    result = {key: copy.deepcopy(val_dict) for key in keys}

    for i in range(1, 4):
        r_sum = 0
        t1_sum = 0
        t2_sum = 0
        for j in range(1, const_count + 1):
            r, t1, t2 = run_inter(i, j, False)
            r_sum = r_sum + r
            t1_sum = t1_sum + t1
            t2_sum = t2_sum + t2
            print('Const_' + str(i) + '.' + str(j) + " done")

        result[rs3 + str(i)][val_r] = r_sum / const_count
        result[rs3 + str(i)][val_t1] = t1_sum / const_count
        result[rs3 + str(i)][val_t2] = t2_sum / const_count