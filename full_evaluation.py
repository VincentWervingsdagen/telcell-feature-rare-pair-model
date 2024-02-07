"""Script containing an example how to use telcell."""
import pickle
import random

from datetime import datetime
from pathlib import Path

from lrbenchmark.evaluation import Setup

from telcell.models.model_count import Count
from telcell.models.model_regression import Regression
from telcell.models.rare_pair_feature_based import RarePairModel

from telcell.data.parsers import parse_measurements_csv
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots, write_lrs
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs, is_colocated


def main():
    """Main function that deals with the whole process. Three steps: loading,
    transforming and evaluation."""

#     eval_files = ["experiments/data/Explorers/Eval/sampling1/output_cell_small.csv",
#                   "experiments/data/Returners/Eval/sampling1/output_cell_small.csv",
#                   "experiments/data/Returners/Eval/sampling2/output_cell_small.csv",
#                   "experiments/data/Returners/Eval/sampling3/output_cell_small.csv",
#                   "experiments/data/CTRW/Eval/output_cell.csv"]
#     file_names = ["explorers1", "returners1","returners2","returners3","movers"]
    eval_files = ["tests/output_cell.csv"]
    file_names = ["explorers1"]

    main_output_dir = Path('scratch')
    all_different_source = False

    # # args for regression
    # train_file1 = "experiments/data/Explorers/Train/sampling1/output_cell_small.csv" # for regression
    # train_file2 = "experiments/data/Returners/Train/sampling1/output_cell_small.csv" # for regression
    # train_file3 = "experiments/data/Returners/Train/sampling2/output_cell_small.csv" # for regression
    # train_file4 = "experiments/data/Returners/Train/sampling3/output_cell_small.csv" # for regression
    # train_file5 = "experiments/data/CTRW/Train/output_cell.csv" # for regression

    # args for regression
    train_file1 = "tests/output_cell.csv" # for regression


    # args for feature based
    bins =  ([0, 0],[1, 20],[21, 40],[41, 60],[61, 120])
    coverage_models = pickle.load(open('coverage_model', 'rb'))

    # args for categorical count
    postcode_file = "data/Postcodevlakken_PC_4.zip" # for categorical count method
    bounding_box = (4.2009,51.8561,4.9423,52.3926)


    # Specify the models that we want to evaluate.
    models = [Regression(parse_measurements_csv(train_file1)),
              Count(postcode_file, bounding_box),
              RarePairModel(bins=bins, coverage_models=coverage_models)]
    
 
    for i,file in enumerate(eval_files):
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
        grid = {'model': models}
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


if __name__ == '__main__':
    main()

