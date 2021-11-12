import sys
import pandas as pd
import time

# setting path
sys.path.append('../src')

import src.post_rec as post_rec

# constants
# file path
input_path = "../../data/"
output_path = '../../result/'

# user/item max
user_max = 1000
food_max = 5000

# constraint sets
const_count = 3


def run_model(ctype, idx):
    start = time.time()

    # PostRec applies constraint after the rating of each item is predicted
    rec = post_rec.PostRec(input_path + 'reduced_rating_data.csv', input_path + 'recipe_data.csv',
                           input_path + 'const_' + str(ctype) + '.' + str(idx) + '.csv')
    rec.get_data()  # get rating, attribute, recipe data
    rec.train()  # train with data
    rec.test()  # predict antisets

    df = rec.get_top_n()  # return result as Dataframe
    df.to_csv(output_path + 'PostRec_' + str(ctype) + '.' + str(idx) + '.csv')

    print('Constraint' + str(ctype) + '.' + str(idx) + " Done, Execution Time(s):", time.time() - start)
    return rec, df


if __name__ == "__main__":
    for i in range(0, const_count):
        rec, df = run_model(3, i + 1)