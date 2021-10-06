import pandas as pd
from surprise import Reader
from surprise import Dataset

# helper function
# cast list of string to list of integer
def int_cast(str_list):
    for i in range (0, len(str_list)):
        if(str_list[i]==''):
            continue
        str_list[i] = int(str_list[i])
    return str_list

# cast list of string to list of float
def float_cast(str_list):
    for i in range (0, len(str_list)):
        if(str_list[i]==''):
            continue
        str_list[i] = float(str_list[i])
    return str_list

# load rating data for surprise
def load_rating_data(file_path='../data/rating_data.csv'):
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 5))
    return Dataset.load_from_file(file_path, reader = reader)

# load recipe data
def load_recipe_data(file_path="../data/recipe_data.csv"):
    df = pd.read_csv(file_path)
    df.set_index('fid', inplace = True)
    
    # parse ingredient list
    df['ingredient_ids'] = df['ingredient_ids'].str.replace(" ", "")
    df['ingredient_ids'] = df['ingredient_ids'].apply(lambda x: x[1:-1].split(','))
    df['ingredient_ids'] = df['ingredient_ids'].apply(lambda x: int_cast(x))
    df['nutrition'] = df['nutrition'].str.replace(" ", "")
    df['nutrition'] = df['nutrition'].apply(lambda x: x[1:-1].split(','))
    df['nutrition'] = df['nutrition'].apply(lambda x: float_cast(x))
    
    return df

# load ingredient related constraint data
def load_ingr_const(file_path="../data/ingr_const.csv"):
    df = pd.read_csv(file_path)
    df.set_index('u', inplace = True)
    
    # parse ingredient list
    df['include'] = df['include'].str.replace(" ", "")
    df['include'] = df['include'].apply(lambda x: x[1:-1].split(','))
    df['include'] = df['include'].apply(lambda x: int_cast(x))
    df['exclude'] = df['exclude'].str.replace(" ", "")
    df['exclude'] = df['exclude'].apply(lambda x: x[1:-1].split(','))
    df['exclude'] = df['exclude'].apply(lambda x: int_cast(x))
    
    return df
