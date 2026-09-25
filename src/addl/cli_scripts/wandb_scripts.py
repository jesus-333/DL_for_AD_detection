"""
Implementation of the cli functions for interaction with wandb

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import argparse
import json
import pandas as pd
import wandb

from addl.other import wandb_interactions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_run_status(args : argparse.Namespace) -> None :
    """
    Given entity, project, and run id return the status of the current run (if exist) and print it in the terminal.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed by the argparse library. The expected arguments are:
        - entity : str, required, the name of the wandb entity (i.e., the user or team name).
        - project : str, required, the name of the wandb project.
        - run_id : str, required, the id of the wandb run.
    """

    # ***************************************
    # Check the arguments

    if args.entity is None  : raise ValueError("The --entity argument is required. Please specify the name of the wandb entity (i.e., the user or team name).")
    if args.project is None : raise ValueError("The --project argument is required. Please specify the name of the wandb project.")
    if args.run_id is None  : raise ValueError("The --run_id argument is required. Please specify the id of the wandb run.")

    # ***************************************

    # Get the run status
    try :
        # Get the run object from wandb using the specified entity, project, and run id
        run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run_id}")
        
        # Print the run status in the terminal
        print(f"{run.state}")
    except wandb.errors.CommError as e :
        print(f"An error occurred while trying to get the run status. Please check the specified arguments and try again. Error message: {str(e)}")
    except Exception as e :
        print(f"An unexpected error occurred while trying to get the runs from the specified project. Please check the specified arguments and try again. Error message: {str(e)}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_project_runs_metrics(args : argparse.Namespace) -> None :
    """
    Given entity and project, download the metrics of all runs in the specified wandb project and save them to a JSON file.

    The output will be a JSON file containing a dictionary where the keys are the metric names and the values are dictionaries of list.
    Each of the internal dictionaries will have the run names as keys and the corresponding metric values as lists.

    Example output :
    {
        "accuracy": {
            run_name_run_id : [0.1, 0.2, 0.3],
            run_name_run_id : [0.15, 0.25, 0.35],
            ...
        },
        "loss": [
            run_name_run_id : [1.0, 0.9, 0.8],
            run_name_run_id : [1.1, 1.0, 0.9],
            ...
        ],
        ...
    }

    """

    # ***************************************
    # Check the arguments

    if args.entity is None  : raise ValueError("The --entity argument is required. Please specify the name of the wandb entity (i.e., the user or team name).")
    if args.project is None : raise ValueError("The --project argument is required. Please specify the name of the wandb project.")

    if args.path_save is None :
        print("The --path_save argument is not specified. The metrics will be saved in the current working directory in the file wandb_metrics.json.")
        path_save = "./wandb_metrics.json"
    else :
        path_save = args.path_save

    # ***************************************
    # Downalod metrics

    try :
        # Get all the runs from the specified project
        runs = wandb.Api().runs(f"{args.entity}/{args.project}")

        metrics_dict = dict()

        for run in runs :
            # Get the run information using the get_run_information function
            run_info = wandb_interactions.get_run_information(run)

            # Get the run history DataFrame and remove the columns that correspond to model parameters
            run_history = run_info['history']
            metrics_df = wandb_interactions.remove_parameters_from_history(run_history)

            # Get run name and id
            name_to_use = f"{run_info['name']}_{run.id}"

            # Save metrics in metrics dict
            for metric in metrics_df.columns :
                # Skip the metrics created by wandb (_step, _runtime, _timestamp)
                if metric.startswith("_") : continue

                # If the metric is not already in the metrics_dict, create a new entry for it
                if metric not in metrics_dict : metrics_dict[metric] = dict()

                # Save the metric values for the current run in the metrics_dict
                metrics_dict[metric][name_to_use] = metrics_df[metric].tolist()

        # Save the metrics_dict to a JSON file
        with open(path_save, 'w') as f : json.dump(metrics_dict, f, indent = 4)

    except wandb.errors.CommError as e :
        print(f"An error occurred while trying to get the runs from the specified project. Please check the specified arguments and try again. Error message: {str(e)}")
    except Exception as e :
        print(f"An unexpected error occurred while trying to get the runs from the specified project. Please check the specified arguments and try again. Error message: {str(e)}")
