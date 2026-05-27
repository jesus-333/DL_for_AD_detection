"""
Implementation of the cli functions for interaction with wandb

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import wandb

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_run_status(args) -> None :
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

    if args.entity is None : raise ValueError("The --entity argument is required. Please specify the name of the wandb entity (i.e., the user or team name).")
    if args.project is None: raise ValueError("The --project argument is required. Please specify the name of the wandb project.")
    if args.run_id is None : raise ValueError("The --run_id argument is required. Please specify the id of the wandb run.")

    # ***************************************

    # Get the run status
    try :
        # Get the run object from wandb using the specified entity, project, and run id
        run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run_id}")
        
        # Print the run status in the terminal
        print(f"{run.state}")
    except wandb.errors.CommError as e :
        print(f"An error occurred while trying to get the run status. Please check the specified arguments and try again. Error message: {str(e)}")

