import glob
import os
import pandas as pd
import itertools
from telcell.utils.savefile import make_output_plots

"""
Code for comination analysis
"""


def aggree(lr1, lr2):
    if (lr1 <= 1 and lr2 <= 1):
        return True 
    elif (lr1 >= 1 and lr2 >= 1):
        return True
    else:
        return False

# main directory where results are stored
main_dir = "scratch/All/movers"
main_dirs = os.listdir(main_dir)

# initialize lists for reading in csv files
all_files = []
column_names = ["Count","RarePairModel","Regression1","Regression2","Regression3","Regression4","Regression5"]
y_true = []

# loop over in each folder in main directory and find and store the lr csv files
for folder in main_dirs:
    if (len(glob.glob(os.path.join(main_dir, folder,"*.csv"))) > 0):
        all_files.append(glob.glob(os.path.join(main_dir, folder,"*.csv"))[0])
        # column_names.append(folder.split("_")[0].replace('movers-model-', ''))

# true values, of same length as lrs
first_file = pd.read_csv(all_files[0])
y_true = list(int(owner_a == owner_b) for (owner_a,owner_b) in zip(first_file["track_a_owner"],first_file["track_b_owner"]))  

# read in all lrs into one dataframe with correct column names
df_lrs = pd.concat((pd.read_csv(f,usecols = ['lr']) for f in all_files), axis=1)
df_lrs.columns = column_names

for a, b in itertools.combinations(["Count","RarePairModel","Regression5"], 2):
    name = a + "x" + b
    # # code for multiplier analysis
    print(name)
    output_dir = os.path.join(main_dir, "multiplier", name)
    mult = list(lr1*lr2 for (lr1,lr2) in zip(df_lrs[a],df_lrs[b]))    
    make_output_plots(mult, y_true, output_dir, ignore_missing_lrs = True)

    # # code for adder analysis
    output_dir = os.path.join(main_dir, "adder", name)
    add = list(lr1+lr2 for (lr1,lr2) in zip(df_lrs[a],df_lrs[b]))    
    make_output_plots(add, y_true, output_dir, ignore_missing_lrs = True)

    # # code for conditional multiplier adder analysis
    output_dir = os.path.join(main_dir, "cond_mult_add", name)
    mult_add = list(lr1*lr2 if aggree(lr1,lr2) else lr1+lr2 for (lr1,lr2) in zip(df_lrs[a],df_lrs[b]))    
    make_output_plots(mult_add, y_true, output_dir, ignore_missing_lrs = True)



