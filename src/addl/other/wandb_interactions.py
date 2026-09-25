"""
Functions to interact with Weights and Biases (wandb) API to retrieve information about runs in a project.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import pandas as pd
import wandb

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_run_information(run : wandb.apis.public.Run) :
    """
    Given a wandb run, return a dictionary with the run's history, summary, config, and name.

    Parameters
    ----------
    run : wandb.apis.public.run
        A wandb run object.

    Returns
    -------
    run_info : dict
        A dictionary containing the run's name, history, summary, and config. 
        - name : str
            The name of the run, as specified in wandb.
        - id : str
            The id of the run, as specified in wandb.
        - history : pandas.DataFrame
            A DataFrame containing the run's history of metrics over time (and model parameters if logged).
        - summary : dict()
            A dictionary containing the run's summary metrics (final values of metrics).
        - config : dict()
            A dictionary containing the run's configuration (hyperparameters), excluding any keys that start with an underscore.
    """
    
    name    = run.name
    id      = run.id
    history = run.history()
    summary = run.summary._json_dict
    config  = {k: v for k, v in run.config.items() if not k.startswith ('_')}

    run_info = dict(
        name    = name,
        id      = id,
        history = history,
        summary = summary,
        config  = config
    )

    return run_info

def remove_parameters_from_history(history_df : pd.DataFrame) -> pd.DataFrame :
    """
    Given a wandb run history DataFrame, remove the columns that correspond to model parameters (i.e., columns that start with "parameters/") and return the modified DataFrame.

    Parameters
    ----------
    history_df : pandas.DataFrame
        A DataFrame containing the run's history of metrics over time.

    Returns
    -------
    history_df_clean : pandas.DataFrame
        A DataFrame containing the run's history of metrics over time, with the columns that correspond to model parameters removed.
    """

    # Get the columns that correspond to model parameters (i.e., columns that start with "parameters/")
    parameter_columns = [col for col in history_df.columns if col.startswith("parameters/")]

    # Drop the parameter columns from the history DataFrame
    history_df_clean = history_df.drop(columns=parameter_columns)

    return history_df_clean

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# import pandas as pd
# import wandb
# api = wandb.Api()
#
# # Project is specified by <entity/project-name>
# runs = api.runs("alberto_zancanaro_academic/demnet_FL_random_partioning_PAPER")
#
# summary_list, config_list, name_list = [], [], []
# for run in runs:
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files
#     summary_list.append(run.summary._json_dict)
#
#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#           if not k.startswith('_')})
#
#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#
# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })
#
# runs_df.to_csv("project.csv")
