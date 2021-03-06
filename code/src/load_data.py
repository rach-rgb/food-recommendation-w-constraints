import numpy as np
import pandas as pd
from surprise import Reader, Dataset

input_path = "../data/"


# helper function
# cast list of string to list of integer
def int_cast(str_list):
    for i in range(0, len(str_list)):
        if str_list[i] == '':
            continue
        str_list[i] = int(str_list[i])
    return str_list


# cast list of string to list of float
def float_cast(str_list):
    for i in range(0, len(str_list)):
        if str_list[i] == '':
            continue
        str_list[i] = float(str_list[i])
    return str_list


# load rating data for surprise
def load_rate(file_path):
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 5))
    return Dataset.load_from_file(file_path, reader=reader)


# load recipe data
def load_attr(file_path):
    df = pd.read_csv(file_path)
    df.set_index('fid', inplace=True)

    # parse ingredient list
    df['ingredient_ids'] = df['ingredient_ids'].str.replace(" ", "")
    df['ingredient_ids'] = df['ingredient_ids'].apply(lambda x: x[1:-1].split(','))
    df['ingredient_ids'] = df['ingredient_ids'].apply(lambda x: int_cast(x))
    df['nutrition'] = df['nutrition'].str.replace(" ", "")
    df['nutrition'] = df['nutrition'].apply(lambda x: x[1:-1].split(','))
    df['nutrition'] = df['nutrition'].apply(lambda x: float_cast(x))

    return df


# load constraint data
def load_const(file_path):
    df = pd.read_csv(file_path)
    df['i1'] = df['i1'].astype('Int64')
    df['i2'] = df['i2'].astype('Int64')
    df = df.fillna(np.nan).replace([np.nan], [None])

    df['nl'] = df['nl'].str.replace(" ", "")
    df['nl'] = df['nl'].apply(lambda x: x[1:-1].split(',') if x is not None else x)
    df['nl'] = df['nl'].apply(lambda x: float_cast(x) if x is not None else x)

    return df
