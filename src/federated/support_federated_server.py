"""
Support functions for the FL module. It contains functions specifically written for server/strategy

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import flwr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import support_federated_generic

try :
    import wandb
    wandb_installed = True
except ImportError :
    print('Warning: wandb is not installed. The class fed_avg_with_wandb_tracking will not works. If you want to use it, please install it using "pip install wandb" ')
    wandb_installed = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def extract_metric_from_log_dict(log_dict : dict) -> (list, list):
    """
    Since (at the time of writing) flower does not allow to save entire array/list inside the log dict I have to save each epoch separately, i.e. with a different entry in the dictionary.
    With this method I will merge all the entry for each specic key and reconvert them to numpy array.

    Parameters
    ----------
        log_dict (dict) :
            Dictionary containing the logs from the server. It must be in the format produced by the function convert_training_metrics_for_upload (see implmentation in the client.py).

    Returns
    -------
        metrics_values_list (list) :
            List of numpy arrays containing the values of the metrics for each epoch. Each element of the list corresponds to a different metric.
        metrics_name_list (list) :
            List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_values_list.
    """
    
    training_epochs = np.arange(log_dict['epochs'])
    
    metrics_values_list = []
    metrics_name_list = []

    # Get the name of the metrics
    for metric_name in log_dict :
        if '0' in metric_name :
            metrics_name_list.append(metric_name.split(":")[0])
    
    # Iterate over possible metrics
    for i in range(len(metrics_name_list)) :
        metric_name = metrics_name_list[i]

        # Create list for the specific metric
        metrics_values_list.append([])

        # Iterate over training epoch
        for j in range(len(training_epochs)) :
            # Get epoch and metric for the epoch
            current_epoch = training_epochs[j]
            metric_for_current_epoch = log_dict[f'{metric_name}:{current_epoch}']

            # Save metric
            metrics_values_list[i].append(metric_for_current_epoch)

        # Convert list to numpy array
        metrics_values_list[i] = np.asarray(metrics_values_list[i])
    
    return metrics_values_list, metrics_name_list

def compute_lr_after_round(lr_value : float, server_round : int) :
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Wandb support functions. Used to setup, aggregate_fit and eveluate functions. 

def setup_wand(wandb_config : dict, config_to_backup : dict) :
    """
    Create the wandb run and the model artifact and return them.

    Parameters
    wandb_config (dict) :
        Dictionary containing the wandb configuration. It must contain the following keys:
            - project_name (str): Name of the wandb project.
            - entity (str, optional): Name of the wandb entity. If not provided, it will be set to None.
            - name_training_run (str, optional): Name of the training run. If not provided, it will be set to None.
            - notes (str, optional): Notes for the training run. If not provided, it will be set to 'No notes in config'.
            - log_model_artifact (bool): Whether to log the model artifact or not.
            - model_artifact_name (str): Name of the model artifact.
            - description (str, optional): Description of the model artifact. If not provided, it will be set to None.
    config_to_backup (dict) :
        Dictionary containing the configuration to backup. It will be saved in the wandb run and as metadata in the model artifact.

    Returns
    -------
    wandb_run (wandb.sdk.wandb_run.Run) :
        The initialized wandb run.
    model_artifact (wandb.sdk.artifacts.artifact) :
        The wandb artifact for the model weights, if logging is enabled. Otherwise, it will be None.
    """

    if not wandb_installed : raise ImportError('wandb is not installed. Please it using "pip install wandb"')
    
    # Get wandb info configuration
    # The extra check here is to avoid KeyError if the wandb_config is not present in the server_config
    # The toml format file does not allow a key with None value. Some of the scripts used to update the config files could set the value in the dictionary to None if the corresponding argument is not provided to the script.
    # In that case the key is not saved in the toml file.
    # To avoid error for this two specific keys, I check if they're present in the wandb_config dictionary.
    # If not I set them to None, since wandb allow None as value for the name and notes of the run.
    entity            = wandb_config['entity'] if 'entity' in wandb_config else None
    name_training_run = wandb_config['name_training_run'] if 'name_training_run' in wandb_config else None
    notes             = wandb_config['notes'] if 'notes' in wandb_config else 'No notes in config'

    # Initialize wandb
    wandb_run = wandb.init(project = wandb_config['project_name'],
                           entity = entity,
                           job_type = "train", config = config_to_backup,
                           name = name_training_run, notes = notes,
                           )

    # Wandb artifact to save model weights
    if wandb_config['log_model_artifact'] :
        model_artifact = wandb.Artifact(wandb_config['model_artifact_name'], type = "model",
                                        description = wandb_config['description'] if 'description' in wandb_config else None,
                                        metadata = config_to_backup)
    else :
        model_artifact = None

    return wandb_run, model_artifact


def aggregate_fit_log(aggregated_parameters, aggregated_metrics, server_round: int, results, model, wandb_run : wandb.sdk.wandb_run.Run, count_rounds : int, rounds_to_save_model : int, num_rounds : int, all_config : dict, model_artifact : wandb.sdk.artifacts.artifact = None, metrics_to_log_from_clients = None) :
    """
    Aggregate the results from the clients and upload the results in wandb.

    Parameters
    ----------
        aggregated_parameters (flwr.common.Parameters) :
            The aggregated parameters from the clients. It is the result of the aggregation function used in the server.
        aggregated_metrics (dict) :
            The aggregated metrics from the clients. It is the result of the aggregation function used in the server.
        server_round (int) :
            The current server round. 
        results (list) :
            List of tuples containing the results from the clients. It is computed automatically by the Flower framework and it is one of the parameters of the aggregate_fit function.
        model :
            The model used in the training. It is the model that has been trained by the clients and aggregated by the server.
        wandb_run (wandb.sdk.wandb_run.Run) :
            The wandb run object to log the results to. It is created by the function setup_wand.
        count_rounds (int) :
            The number of rounds executed so far. It is used to log the results in wandb and to save the model every rounds_to_save_model.
        rounds_to_save_model (int) :
            The number of rounds after which the model weights are saved. If it is set to 0 or a negative number, the model weights are not saved.
        num_rounds (int) :
            The total number of rounds to execute. It is used to check if the training is finished.
        all_config (dict) :
            The entire configuration dictionary. It is used to get the path to save the model and other configuration parameters.
            Note that only 2 values inside the dictionary are used in this function. But since this function was originally created inside the server class I extract the data that I need from the all_config dictionary (that is a parameter of the server class).
            To avoid a further increse of the nubmer of parameters of this function, I keep the all_config parameter. Note also the in the end this function is still called by the server class, so it is not a problem to get the entire configuration dictionary.
        model_artifact (wandb.sdk.artifacts.artifact, optional) :
            The wandb artifact for the model weights, if logging is enabled. Otherwise, it will be None.
            If provided, it will be logged in the wandb run.
        metrics_to_log_from_clients (list/str, optional) :
            The metrics to log from the clients. It can be a list of strings with the names of the metrics to log or 'all' to log all the metrics.
            If it is None, no metrics are logged from the clients.
            If it is 'all', all the metrics are logged from the clients.
            If it is a list, only the metrics in the list are logged from the clients.

    Returns
    -------
        aggregated_parameters (flwr.common.Parameters) :
            The aggregated parameters from the clients. It is the result of the aggregation function used in the server.
        aggregated_metrics (dict) :
            The aggregated metrics from the clients. It is the result of the aggregation function used in the server.
        model_weights :
            The weights of the model obtaiend by the aggregated_parameters through the function flwr.common.parameters_to_ndarrays.
        count_rounds (int) :
            The number of rounds executed so far. It is incremented by 1 at the end of the function.
            It is returned so that it can be updated in the server class (I know, I can simply increment it in the server class, but I like the idea of increasing it here and returning it to the server class).
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Clients metrics logging

    if aggregated_parameters is not None:
        model_weights = aggregate_fit_log_clients_data(aggregated_parameters, model, count_rounds, rounds_to_save_model, all_config, results, wandb_run, model_artifact, metrics_to_log_from_clients)
    else :
        model_weights = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Server side metrics logging

    if aggregated_metrics is not None :
        aggregate_fit_log_server_metrics(aggregated_metrics, wandb_run, count_rounds)
    else :
        print("Warning. No aggregated metrics obtained during aggregation phase. fit_metrics_aggregation_fn must is None or there are error with the function code.")
        # print("Only the aggregated weights and the metrics of the clients will be uploaded ")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final operations of the round
    
    # Increase the number of executed rounds
    count_rounds += 1

    # Check If I finished the FL training
    if count_rounds == num_rounds :
        
        # Save model at the end of the training
        save_path = support_federated_generic.save_model_weights(model, path_to_save_model = all_config['training_config']['path_to_save_model'], filename = "model_round_END.pth")
        
        # (OPTIONAL) Upload the model in wandb
        if model_artifact is not None :
            model_artifact.add_file(save_path)
            wandb.save(save_path)
        
        # If I did not have any evaluation function conclude the run
        # Otherwise, if there is a central evaluation function the run is concluded inside evaluate()
        if not all_config['server_config']['centralized_evaluation'] :
            end_wandb_run_and_log_artifact()
            print("End training rounds")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    return aggregated_parameters, aggregated_metrics, model_weights, count_rounds

def aggregate_fit_log_clients_data(aggregated_parameters, model, count_rounds : int, rounds_to_save_model : int, all_config : dict, results : list, wandb_run : wandb.sdk.wandb_run.Run, model_artifact : wandb.sdk.artifacts.artifact = None, metrics_to_log_from_clients = None) -> (flwr.common.Parameters, list):
    """
    Aggregate the parameters from the clients and log the clients metrics in wandb.
    For clients metrics, I mean the metrics computed at each epoch for each clients (e.g. the training loss at each epoch for each client).
    """
    # Convert `Parameters` to `List[np.ndarray]`
    model_weights = flwr.common.parameters_to_ndarrays(aggregated_parameters)

    # Load updated weights into the model
    support_federated_generic.set_weights(model, model_weights)

    # Save weights every rounds_to_save_model
    if count_rounds % rounds_to_save_model == 0 and rounds_to_save_model > 0:
        save_path = support_federated_generic.save_model_weights(model, path_to_save_model = all_config['training_config']['path_to_save_model'], filename = f"model_round_{count_rounds}.pth")
        
        # Upload the model in wandb
        # N.b. The self.model_artifact is None only if the parameter log_model_artifact in wandb config is False
        if model_artifact is not None :
            model_artifact.add_file(save_path)
            wandb.save(save_path)

    if metrics_to_log_from_clients is not None :
        # Iterate over clients
        for i in range(len(results)) :
            # Get metrics log dict for current client
            log_dict = results[i][1].metrics
            
            # Get id and number of training epoch
            client_id = log_dict['client_id']

            # Create epoch arrays
            training_epochs = np.arange(log_dict['epochs']) + 1

            # Extract losses
            metrics_values_list, metrics_name_list = extract_metric_from_log_dict(log_dict)

            # Plot(s) creation and log
            if metrics_to_log_from_clients == 'all' :
                create_and_log_wandb_metric_plot_separately(metrics_values_list, training_epochs, metrics_name_list, client_id, count_rounds, wandb_run)
            else :
                # Get the metrics to plot
                idx_of_metrics_to_plot = [i for i in range(len(metrics_values_list)) if metrics_name_list[i] in metrics_to_log_from_clients]
                metrics_values_to_plot_list = [metrics_values_list[idx] for idx in idx_of_metrics_to_plot]
                metrics_name_to_plot_list   = [metrics_name_list[idx] for idx in idx_of_metrics_to_plot]
                
                # TODO add option to decide if upload also the plot with the metrics merged
                create_and_log_wandb_metric_plot_separately(metrics_values_to_plot_list, training_epochs, metrics_name_to_plot_list, client_id, count_rounds, wandb_run)

                # For now this not produce the plot I want
                # self.create_and_log_wandb_metric_plot_together(metrics_values_to_plot_list, training_epochs, metrics_name_to_plot_list, client_id)

            # for metric_values, metric_name in zip(metrics_values_list, metrics_name_list) :
            #     self.create_and_log_matplotlib_metric_plot([metric_values], training_epochs, [metric_name], client_id)
            #     self.create_and_log_wandb_metric_plot_separately([metric_values], training_epochs, [metric_name], client_id)

    return model_weights

def aggregate_fit_log_server_metrics(aggregated_metrics : dict, wandb_run : wandb.sdk.wandb_run.Run, count_rounds : int) -> None:
    """
    Check if the aggregated metrics are not None and upload them in wandb.
    Note that this are not the entire metrics computed at each epoch for each client. This metrics are the final values of each metric for each client aggregated at the end of the round.
    The aggregation depends on the type of aggregation function used in the server.
    """

    # Dictionary to upload
    wandb_log_dict = dict()
    
    # Extract the metric that I want to upload
    for metric_name in aggregated_metrics :
        # Check if it is the metric at the last iteration.
        # See the function convert_training_metrics_for_upload in the client.py file for more information.
        if 'END' in metric_name :
            # Save the value of the metric
            wandb_log_dict[metric_name.split(":")[0]] = aggregated_metrics[metric_name]
    
    # Upload metric
    print("FIT ", count_rounds)
    wandb_run.log(wandb_log_dict, step = count_rounds)
    # self.wandb_run.log(wandb_log_dict)

def evaluate_log(test_metrics_dict : dict, wandb_run : wandb.sdk.wandb_run.Run, count_rounds : int, num_rounds : int, model_artifact : wandb.sdk.artifacts.artifact = None) -> (float, dict):
    """
    After the centralized evaluation of the model, log the metrics in wandb and close the run if it is the last round.
    """

    # Upload metrics
    wandb_run.log(test_metrics_dict, step = count_rounds)

    # Close wandb run if I'm at the last round
    if count_rounds == num_rounds :
        end_wandb_run_and_log_artifact()
        print("End training rounds")

def end_wandb_run_and_log_artifact(wandb_run : wandb.sdk.wandb_run.Run, model_artifact : wandb.sdk.artifacts.artifact = None) :
    """
    Load the artifacts in wandb and finish the run.

    Parameters
    ----------
        wandb_run (wandb.sdk.wandb_run.Run) :
            The wandb run object to finish. It is created by the function setup_wand.
        model_artifact (wandb.sdk.artifacts.artifact, optional) :
            The wandb artifact for the model weights, if logging is enabled. Otherwise, it will be None.
            If provided, it will be logged in the wandb run.
    """

    # (OPTIONAL) Log model artifact
    if model_artifact is not None : wandb_run.log_artifact(model_artifact)

    # Conclude run
    wandb_run.finish()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Wandb support functions. Used to create and log plots of the metrics.

def create_matplotlib_metric_plot(metrics_to_plot_list : list, training_epochs , metrics_name_list : list) :
    """
    Create a plot with matplotlib for the metrics in metrics_to_plot_list. Each element of the list must be an array/list with the value of the metric for each epoch.

    Parameters
    ----------
        metrics_to_plot_list (list) :
            List of lists/arrays containing the values of the metrics to plot. Each element of the list corresponds to a different metric.
        training_epochs (list) :
            List of epochs for which the metrics are computed.
        metrics_name_list (list) :
            List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_to_plot_list.
    """
    
    # Apparently matplotlib is not thread safe.
    # The creation of fig and ax with the command plt.subplots() without using the default backend will cause the following warning and the crash of python.
    # UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
    # A possible solution suggested online was the use of a non-interactive backend
    matplotlib.use('agg')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = (16, 10))
    fontsize = 16

    for i in range(len(metrics_name_list)) :
        # Plot the metric
        ax.plot(training_epochs, metrics_to_plot_list[i], label = metrics_name_list[i])
        
        # Add details
        ax.legend(fontsize = fontsize)
        ax.set_xlabel('Epoch', fontsize = fontsize)
        ax.set_ylabel('Loss', fontsize = fontsize)
        ax.grid(True)

    fig.tight_layout()

    return fig, ax

def create_and_log_matplotlib_metric_plot(metrics_to_plot_list : list, training_epochs, metrics_name_list, client_id, count_rounds : int, wandb_run) :
    """
    Create a plot with matplotlib for the metric(s) inside metrics_to_plot_list and log it to wandb.

    Parameters
    ----------
        metrics_to_plot_list (list) :
            List of lists/arrays containing the values of the metrics to plot. Each element of the list corresponds to a different metric.
        training_epochs (list) :
            List of epochs for which the metrics are computed.
        metrics_name_list (list) :
            List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_to_plot_list.
        client_id (int/str) :
            ID of the client for which the metrics are computed. It is used to create the name of the plot in wandb.
        count_rounds (int) :
            Number of the current round. It is used to create the name of the plot in wandb.
        wandb_run (wandb.sdk.wandb_run.Run) :
            The wandb run object to log the plot to. It is created by the function setup_wand.
    """
    fig, _ = create_matplotlib_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id)
    
    # Name for the plot to log
    if len(metrics_name_list) == 1 :
        # Only 1 metric
        metric_name = metrics_name_list[0]
    else :
        # More than 1 metric
        metric_name = 'all'
    
    # Log the plot in wandb
    wandb_run.log({
        f"client_{client_id}/{metric_name}_round_{count_rounds}_plt" : fig
    }, commit = False)

    plt.close()

def create_and_log_wandb_metric_plot_separately(metrics_to_plot_list, training_epochs, metrics_name_list, client_id, count_rounds : int, wandb_run) :
    """
    Similar to create_and_log_matplotlib_metric_plot but the plot is created through wandb functions.
    Note that this function create a single plot for every metric you want to log. If you want to log more metrics in a single plot use create_and_log_wandb_metric_plot_together
    Here more info https://docs.wandb.ai/guides/track/log/plots/ and https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA

    Parameters
    ----------
        metrics_to_plot_list (list) :
            List of lists/arrays containing the values of the metrics to plot. Each element of the list corresponds to a different metric.
        training_epochs (list) :
            List of epochs for which the metrics are computed.
        metrics_name_list (list) :
            List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_to_plot_list.
        client_id (int/str) :
            ID of the client for which the metrics are computed. It is used to create the name of the plot in wandb.
        count_rounds (int) :
            Number of the current round. It is used to create the name of the plot in wandb.
        wandb_run (wandb.sdk.wandb_run.Run) :
            The wandb run object to log the plot to. It is created by the function setup_wand.
    """
    
    for i in range(len(metrics_to_plot_list)) :
        metric = metrics_to_plot_list[i]
        metric_name = metrics_name_list[i]

        # Convert the metric for wandb upload
        data = [[current_epoch, metric_value_current_epoch] for (current_epoch, metric_value_current_epoch) in zip(range(len(metric)), metric)]

        # Create the table
        table = wandb.Table(data = data, columns=["epochs", metric_name])

        # Log the plot in wandb
        wandb_run.log({
            f"client_{client_id}/{metric_name}_round_{count_rounds}" : wandb.plot.line(
                table, "epochs", metric_name, title = f"{metric_name}_round_{count_rounds}"
            )
        }, commit = False)
        
        # Note on commit parameter
        # Description of commit parameter (from https://docs.wandb.ai/ref/python/log/)
        #   commit : If true, finalize and upload the step. If false, then accumulate data for the step. See the notes in the description. If step is None, then the default is commit=True; otherwise, the default is commit=False.
        # By default also step is None. So by default commit is True
        
        # Note on the / symbol inside the name of the object logged.
        # If I had the / symbol in the name all the log will be grouped together in a new section with that symbol.
        # E.g. if I log client_1/train_loss and client_1/test_loss I will create two plots called train_loss and test_loss in a specific NEW section called client_1

def create_and_log_wandb_metric_plot_together(metrics_to_plot_list, training_epochs, metrics_name_list, client_id, count_rounds : int, wandb_run) :
    """
    Similar to create_and_log_matplotlib_metric_plot_separately but the metrics are plotted in the same figure.

    Parameters
    ----------
        metrics_to_plot_list (list) :
            List of lists/arrays containing the values of the metrics to plot. Each element of the list corresponds to a different metric.
        training_epochs (list) :
            List of epochs for which the metrics are computed.
        metrics_name_list (list) :
            List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_to_plot_list.
        client_id (int/str) :
            ID of the client for which the metrics are computed. It is used to create the name of the plot in wandb.
        count_rounds (int) :
            Number of the current round. It is used to create the name of the plot in wandb.
        wandb_run (wandb.sdk.wandb_run.Run) :
            The wandb run object to log the plot to. It is created by the function setup_wand.
    """
    
    # Create the first columns with epochs
    data = [[epoch] for epoch in training_epochs]

    # Add the other metrics as columns
    plot_log_name = f"client_{client_id}/"
    for i in range(len(metrics_to_plot_list)) :
        metric_array = metrics_to_plot_list[i]
        metric_name = metrics_name_list[i]
        plot_log_name += f"{metric_name}_"
        
        # Iterate over epoch (i.e. get the value of the current metric for every epoch)
        for j in range(len(metric_array)) : data[j].append(metric_array[j])
    
    plot_log_name += f"round_{count_rounds}"

    # Create the table 
    table = wandb.Table(data = data, columns = ["epochs", *metrics_name_list])

    # Log the plot in wandb
    wandb_run.log({
        plot_log_name : wandb.plot.line(
            table, "epochs", "Metrics value", title = f"{[metrics_name_list]}_round_{count_rounds}"
        )
    }, commit = False)


