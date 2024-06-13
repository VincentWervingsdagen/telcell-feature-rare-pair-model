"""Script containing an example how to use telcell."""
import pickle
import random

from datetime import datetime
from pathlib import Path

from lrbenchmark.evaluation import Setup

from telcell.models.model_count import Count
from telcell.models.model_regression import Regression
from telcell.models.rare_pair_feature_based import RarePairModel
from telcell.models.markov_model import MarkovChain

from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import check_header_for_postal_code, add_postal_code
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots, write_lrs
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs, is_colocated


def main():
    """Main function that deals with the whole process. Three steps: loading,
    transforming and evaluation."""

    scenario = 'baseline'
    test_files = "data/{}/test_set_{}.csv".format(scenario,scenario)
    train_files = "data/{}/training_set_{}.csv".format(scenario,scenario)
    file_names = [scenario]

    all_different_source = False

    # args for regression

    # args for feature based
    bins =  ([0, 0],[1, 20],[21, 40],[41, 60],[61, 120])
    coverage_models = pickle.load(open('data/coverage_model', 'rb'))

    # args for categorical count
    postcode_file = "data/Postcodevlakken_PC_4.zip" # for categorical count method
    bounding_box = (4.2009,51.8561,4.9423,52.3926)

    # args for Markov chain approach
    state_space = 'observations'
    state_space_level = 'postal2'
    distance = 'frobenius'


    # Check whether the files have 'cellinfo.postal_code' column.
    for file in [train_files,test_files]: # Makes sure that the column cellinfo.postal_code is available
        if check_header_for_postal_code(file):
            pass
        else:
            add_postal_code(file)

    #Specify the models that we want to evaluate.
    models_days = [Regression(parse_measurements_csv(train_files)),
              RarePairModel(bins=bins, coverage_models=coverage_models)]

    models_period = [
                     Count(postcode_file, bounding_box),
                     MarkovChain(training_set=train_files,cell_file=coverage_models,bounding_box=bounding_box,
                                 state_space=state_space,state_space_level=state_space_level,distance=distance)]

    # Loading data
    tracks = parse_measurements_csv(file)
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
            unique_dir = file_names[i]+'-'+'_'.join(f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
                    "%Y-%m-%d %H_%M_%S")
            output_dir = main_output_dir / unique_dir
            make_output_plots(predicted_lrs,
                            y_true,
                            output_dir,
                            ignore_missing_lrs=True)

    # Split the data in period of half a year.
    data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=all_different_source),interval_length_h=4464))
    setup.parameter('data', data)

    grid = {'model': models_period}
    for variable, parameters, (predicted_lrs, y_true, extras) in \
            setup.run_full_grid(grid):
        model_name = parameters['model'].__class__.__name__
        print(f"{model_name}: {predicted_lrs}")
        unique_dir = file_names[i] + '-' + '_'.join(
            f'{key}-{value}' for key, value in variable.items()) + '_' + datetime.now().strftime(
            "%Y-%m-%d %H_%M_%S")
        output_dir = main_output_dir / unique_dir
        make_output_plots(predicted_lrs,
                          y_true,
                          output_dir,
                          ignore_missing_lrs=True)


if __name__ == '__main__':
    main()

