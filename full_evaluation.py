"""Script containing an example how to use telcell."""
import pickle
import random

from datetime import datetime
from pathlib import Path

from lrbenchmark.evaluation import Setup

from telcell.models.model_count import Count
from telcell.models.model_regression import Regression
from telcell.models.rare_pair_feature_based import RarePairModel
from telcell.models.model_markov import MarkovChain

from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import check_header_for_postal_code, add_postal_code
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots, write_lrs
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs, is_colocated


def main():
    """Main function that deals with the whole process. Three steps: loading,
    transforming and evaluation."""

    scenario = 'common_work'
    test_files = "data/Vincent/{}/test_set_{}.csv".format(scenario,scenario)
    train_files = "data/Vincent/{}/training_set_{}.csv".format(scenario,scenario)

    all_different_source = False

    # args for regression

    # args for feature based
    bins =  ([0, 0],[1, 20],[21, 40],[41, 60],[61, 120])
    coverage_models = pickle.load(open('data/coverage_model', 'rb'))

    # args for categorical count
    bounding_box = (4.2009,51.8561,4.9423,52.3926)

    # args for Markov chain approach 1
    cell_file = "tests/20191202131001.csv"
    markov_1 = ['Omega', 'postal2', 'frobenius']

    # args for Markov chain approach 2
    markov_2 = ['Omega', 'postal3', 'important_cut_distance']

    # Check whether the files have 'cellinfo.postal_code' column.
    for file in [train_files,test_files]: # Makes sure that the column cellinfo.postal_code is available
        if check_header_for_postal_code(file):
            pass
        else:
            add_postal_code(file)

    #Specify the models that we want to evaluate.
    models_days = [
                   RarePairModel(bins=bins, coverage_models=coverage_models),
                   Count()
    ]

    models_period = [
                     MarkovChain(training_set=train_files,cell_file=cell_file,bounding_box=bounding_box,
                                 state_space=markov_1[0],state_space_level=markov_1[1],distance=markov_1[2]),
                     MarkovChain(training_set=train_files, cell_file=cell_file, bounding_box=bounding_box,
                                 state_space=markov_2[0], state_space_level=markov_2[1], distance=markov_2[2]),
                     Count(),
    ]

    # Loading data
    tracks = parse_measurements_csv(test_files)

    data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=all_different_source),interval_length_h=24))

    # Sample negative pairs
    colocated, not_colocated = [], []
    for row in data:
            (not_colocated,colocated)[is_colocated(row[0],row[1])].append(row)
    not_colocated = random.sample(not_colocated, k=len(colocated))
    data = colocated + not_colocated

    # Create an experiment setup using run_pipeline as the evaluation function
    setup = Setup(run_pipeline)

    # Specify the constant parameters for evaluation
    setup.parameter('data', data)
    setup.parameter('filter', {'mnc': ['8', '16']})

    # Specify the main output_dir. Each model/parameter combination gets a
    # directory in the main output directory.
    main_output_dir = Path('scratch')

    # Specify the variable parameters for evaluation in the variable 'grid'.
    # This grid is a dict of iterables and all combinations will be used
    # during the evaluation. An example is a list of all different models
    # that need to be evaluated, or a list of different parameter settings
    # for the models.
    grid = {'model': models_days}
    for variable, parameters, (predicted_lrs, y_true, extras) in \
            setup.run_full_grid(grid):
            model_name = parameters['model'].__class__.__name__
            print(f"{model_name}: {predicted_lrs}")
            unique_dir = scenario+'-'+'_'.join(f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
                    "%Y-%m-%d %H_%M_%S")
            output_dir = main_output_dir / unique_dir
            make_output_plots(predicted_lrs,
                            y_true,
                            output_dir,
                            ignore_missing_lrs=True)



    # Split the data in period of half a year.
    data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=all_different_source),interval_length_h=4464))
    colocated, not_colocated = [], []
    for row in data:
            (not_colocated,colocated)[is_colocated(row[0],row[1])].append(row)
    not_colocated = random.sample(not_colocated, k=5*len(colocated))
    data = colocated + not_colocated

    # Create an experiment setup using run_pipeline as the evaluation function
    setup = Setup(run_pipeline)

    # Specify the constant parameters for evaluation
    setup.parameter('filter', {'mnc': ['8', '16']})
    setup.parameter('data', data)

    # Specify the main output_dir. Each model/parameter combination gets a
    # directory in the main output directory.
    main_output_dir = Path('scratch')

    grid = {'model': models_period}
    for variable, parameters, (predicted_lrs, y_true, extras) in \
            setup.run_full_grid(grid):
        model_name = parameters['model'].__class__.__name__
        print(f"{model_name}: {predicted_lrs}")
        unique_dir = scenario + '-' + '_'.join(
            f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
            "%Y-%m-%d %H_%M_%S")
        output_dir = main_output_dir / unique_dir
        make_output_plots(predicted_lrs,
                          y_true,
                          output_dir,
                          ignore_missing_lrs=True)


if __name__ == '__main__':
    main()

