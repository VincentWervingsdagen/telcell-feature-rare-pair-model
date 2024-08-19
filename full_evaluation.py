"""Script containing an example how to use telcell."""
import pickle
import random
import os
from datetime import datetime
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from lrbenchmark.evaluation import Setup
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from telcell.models.model_count import Count
from telcell.models.model_count_ELUB import Count_ELUB
from telcell.models.model_regression import Regression
from telcell.models.rare_pair_feature_based import RarePairModel
from telcell.models.model_markov import MarkovChain

from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import check_header_for_postal_code, add_postal_code, calculate_bounding_box
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots, write_lrs
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs, is_colocated


def main():
    """Main function that deals with the whole process. Three steps: loading,
    transforming and evaluation."""

    # Edit these variables
    scenario = 'baseline' # Scenario name to create an appropriate file path.
    output_cell_file = "data/Vincent/{}/output_cell.csv".format(scenario) # The csv file with all observations.

    # Leave the rest as it is
    response_ELUB = input(
        f"Do you want to use an ELUB-bounder for the Markov chain? Our advise is to use an ELUB-bounder. (y/n): ").strip().lower()
    response_H_d_pairs = input(f"Do you want all possible H_d pairs? All H_d pairs might take a long time to run, "
                               f"so otherwise a sample of 5 times the number of the H_p instead. (y/n): ").strip().lower()
    response_cross_validation = input(f"Do you want to use cross validation with five folds? "
        f"Otherwise a training test split of 80/20 percent will be used. (y/n): ").strip().lower()

    # Specify the main output_dir. Each model/parameter combination gets a
    # directory in the main output directory.
    main_output_dir = Path(f'scratch/{scenario}')

    all_different_source = False

    # args for regression

    # args for feature based
    bins = ([0, 0], [1, 20], [21, 40], [41, 60], [61, 120])
    coverage_models = pickle.load(open('data/coverage_model', 'rb'))

    # args for categorical count
    bounding_box = calculate_bounding_box(output_cell_file)

    # args for Markov chain approach 1
    cell_file = "tests/20191202131001.csv"
    markov_frobenius = ['postal2', 'Frobenius_norm','uniform']

    # args for Markov chain approach 2
    markov_cut_distance = ['postal3', 'cut_distance','overall objective']

    # Check whether the files have 'cellinfo.postal_code' column.
    for file in [output_cell_file]:  # Makes sure that the column cellinfo.postal_code is available
        if check_header_for_postal_code(file):
            pass
        else:
            add_postal_code(file)

    df_output_cell = pd.read_csv(output_cell_file)

    if response_cross_validation == 'y':
        dataset = GroupKFold(n_splits=5).split(X=df_output_cell,groups=df_output_cell['owner'])
    else:
        dataset = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=40).split(X=df_output_cell,groups=df_output_cell['owner'])

    df_results = pd.DataFrame()
    fold_number = 0
    for train_index, test_index in dataset:
        data_train = df_output_cell.iloc[train_index]
        data_test = df_output_cell.iloc[test_index]
        fold_number += 1
        predicted_lr_list = dict()
        output_dir_fold = main_output_dir / f"fold_{fold_number}"

        #Specify the models that we want to evaluate.
        # models_days = [
        #                RarePairModel(bins=bins, coverage_models=coverage_models),
        #                Count()
        # ]

        models_period = [
                         MarkovChain(df_train=data_train,cell_file=cell_file,bounding_box=bounding_box,
                                     output_histogram_path=output_dir_fold,response_ELUB=response_ELUB,
                                     state_space_level=markov_frobenius[0],distance=markov_frobenius[1],prior_type=markov_frobenius[2]),
                         MarkovChain(df_train=data_train, cell_file=cell_file, bounding_box=bounding_box,
                                    output_histogram_path=output_dir_fold,response_ELUB=response_ELUB,
                                     state_space_level=markov_cut_distance[0], distance=markov_cut_distance[1],prior_type=markov_cut_distance[2]),
                         Count_ELUB(training_set=data_train),
        ]

        # Loading data
        data_test.to_csv(main_output_dir/"bin") # Ugly solution, but the track functions are found in the other methods and only accept csv files. So will leave it as is.
        tracks = parse_measurements_csv(main_output_dir/"bin")

        # # This part of the code is used for the single day methods.
        # data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=all_different_source),interval_length_h=24))
        #
        # # Sample negative pairs
        # colocated, not_colocated = [], []
        # for row in data:
        #         (not_colocated,colocated)[is_colocated(row[0],row[1])].append(row)
        # if response_H_d_pairs == 'y':
        #     pass
        # else:
        #     not_colocated = random.sample(not_colocated, k=len(colocated))
        # data = colocated + not_colocated
        #
        # # Create an experiment setup using run_pipeline as the evaluation function
        # setup = Setup(run_pipeline)
        #
        # # Specify the constant parameters for evaluation
        # setup.parameter('data', data)
        # setup.parameter('filter', {'mnc': ['8', '16']})
        #
        # # Specify the variable parameters for evaluation in the variable 'grid'.
        # # This grid is a dict of iterables and all combinations will be used
        # # during the evaluation. An example is a list of all different models
        # # that need to be evaluated, or a list of different parameter settings
        # # for the models.
        #
        # grid = {'model': models_days}
        # for variable, parameters, (predicted_lrs, y_true, extras) in \
        #         setup.run_full_grid(grid):
        #         model_name = parameters['model'].__class__.__name__
        #         print(f"{model_name}: {predicted_lrs}")
        #         unique_dir = '_'.join(f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
        #                 "%Y-%m-%d %H_%M_%S")
        #         output_dir = output_dir_fold / unique_dir
        #         make_output_plots(predicted_lrs,
        #                         y_true,
        #                         output_dir,
        #                         ignore_missing_lrs=True)

        # This part of the code is used for longer periods.
        # Split the data in periods of half a year.
        data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=all_different_source),interval_length_h=4500))
        colocated, not_colocated = [], []
        for row in data:
                (not_colocated,colocated)[is_colocated(row[0],row[1])].append(row)
        if response_H_d_pairs == 'y':
            pass
        else:
            not_colocated = random.sample(not_colocated, k=5 * len(colocated))
        data = colocated + not_colocated

        # Create an experiment setup using run_pipeline as the evaluation function
        setup = Setup(run_pipeline)

        # Specify the constant parameters for evaluation
        setup.parameter('filter', {'mnc': ['8', '16']})
        setup.parameter('data', data)

        grid = {'model': models_period}

        for variable, parameters, (predicted_lrs, y_true, extras) in \
                setup.run_full_grid(grid):
            model_name = parameters['model'].__class__.__name__

            if model_name == 'MarkovChain':
                model_name = model_name + "_" + parameters['model'].get_distance()
                print(f"{model_name}: {predicted_lrs}")
                unique_dir = f"{model_name}" + datetime.now().strftime(
                    "%Y-%m-%d %H_%M_%S")
            else:
                print(f"{model_name}: {predicted_lrs}")
                unique_dir = f"{model_name}-" + datetime.now().strftime(
                    "%Y-%m-%d %H_%M_%S")

            output_dir = output_dir_fold / unique_dir
            make_output_plots(predicted_lrs,
                              y_true,
                              bounds=None,
                              output_dir=output_dir,
                              ignore_missing_lrs=False)
            predicted_lr_list[model_name] = predicted_lrs


        predicted_lr_list['y_true'] = y_true
        if df_results.empty:
            df_results = pd.DataFrame(predicted_lr_list)
        else:
            df_results = pd.concat([df_results,pd.DataFrame(predicted_lr_list)],ignore_index=True)

    if response_cross_validation == 'y':
        bounds = (df_results.loc[:,df_results.columns!='y_true'].min().min(),df_results.loc[:,df_results.columns!='y_true'].max().max())
        print(bounds)
        for model in [col for col in df_results.columns if col != 'y_true']:
            output_dir = main_output_dir / "total" / model
            make_output_plots(df_results[model],df_results['y_true'],bounds,output_dir,ignore_missing_lrs=False)
    else:
        pass

    file_to_delete = main_output_dir/"bin"
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' successfully deleted.")
        except OSError as e:
            print(f"Error deleting file '{file_to_delete}': {e}")
    else:
        print(f"File '{file_to_delete}' does not exist.")

if __name__ == '__main__':
    main()

