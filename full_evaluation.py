"""Script containing an example how to use telcell."""
import pickle
import random

from datetime import datetime
from pathlib import Path

from lrbenchmark.evaluation import Setup

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

    train = 'common_work'
    test = 'common_work'

    response_ELUB = input(
        f"Do you want to use an ELUB-bounder for the Markov chain? Our advise is to use an ELUB-bounder. (y/n): ").strip().lower()
    response_H_d_pairs = input(f"Do you want all possible H_d pairs? All H_d pairs might take a long time to run, "
                               f"so otherwise a sample of 5 times the number of the H_p instead. (y/n): ").strip().lower()

    # Specify the main output_dir. Each model/parameter combination gets a
    # directory in the main output directory.
    main_output_dir = Path(f'scratch/test2/{train}-{test}')


    train_files = "data/Vincent/{}/training_set_{}.csv".format(train,train)
    test_files = "data/Vincent/{}/test_set_{}.csv".format(test,test)

    all_different_source = False

    # args for regression

    # args for feature based
    bins =  ([0, 0],[1, 20],[21, 40],[41, 60],[61, 120])
    coverage_models = pickle.load(open('data/coverage_model', 'rb'))

    # args for categorical count
    bounding_box = calculate_bounding_box(train_files,test_files)

    # args for Markov chain approach 1
    cell_file = "tests/20191202131001.csv"
    markov_1 = ['postal2', 'frobenius','all_ones']

    # args for Markov chain approach 2
    markov_2 = ['postal3', 'important_cut_distance_5','jeffrey']

    # args for Markov chain approach 3
    markov_3 = ['postal2', 'GLR', 'jeffrey']

    # Check whether the files have 'cellinfo.postal_code' column.
    for file in [train_files,test_files]: # Makes sure that the column cellinfo.postal_code is available
        if check_header_for_postal_code(file):
            pass
        else:
            add_postal_code(file)

    #Specify the models that we want to evaluate.
    # models_days = [
    #                RarePairModel(bins=bins, coverage_models=coverage_models),
    #                Count()
    # ]

    models_period = [
                     MarkovChain(training_set=train_files,cell_file=cell_file,bounding_box=bounding_box,
                                 output_histogram_path=main_output_dir,response_ELUB=response_ELUB,
                                 state_space_level=markov_1[0],distance=markov_1[1],prior_type=markov_1[2]),
                     MarkovChain(training_set=train_files, cell_file=cell_file, bounding_box=bounding_box,
                                output_histogram_path=main_output_dir,response_ELUB=response_ELUB,
                                 state_space_level=markov_2[0], distance=markov_2[1],prior_type=markov_2[2]),
                     MarkovChain(training_set=train_files, cell_file=cell_file, bounding_box=bounding_box,
                                   output_histogram_path=main_output_dir,response_ELUB=response_ELUB,
                                 state_space_level=markov_3[0], distance=markov_3[1],prior_type=markov_3[2]),
                     Count_ELUB(training_set=train_files),
    ]

    # Loading data
    tracks = parse_measurements_csv(test_files)

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
    #         output_dir = main_output_dir / unique_dir
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
        print(f"{model_name}: {predicted_lrs}")
        unique_dir = '_'.join(f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
            "%Y-%m-%d %H_%M_%S")
        output_dir = main_output_dir / unique_dir
        make_output_plots(predicted_lrs,
                          y_true,
                          output_dir,
                          ignore_missing_lrs=True)



if __name__ == '__main__':
    main()

