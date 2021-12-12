# Food Recommendation System with Constraints
Postech CSED499I-01 source code and samples

## datasets
source: [Food.com dataset](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)  
Rating data: project > data > reduced_rating_data.csv  
Attribute data: project > data > recipe_data.csv  
Constraint data:  
[Single constraint] project > data > const_A.B.csv  
A is constraint type and B denotes version of dataset  
[Mixed constraint] project > data > const.B.csv  
B denotes version of dataset

## notebooks
Input path: project > data  
Ouput path: project > result

## cython compilation
svd_constraint.pyx use cython for SGD. Execute `python setup.py build_ext --inplace` at project > code > src and move generated win_amd64 file to src folder.
