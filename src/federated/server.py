"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import flwr
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

try : 
    import wandb
    wandb_installed = True
except ImportError :
    print('Warning: wandb is not installed. The class fed_avg_with_wandb_tracking will not works. If you want to use it, please install it using "pip install wandb" ')
    wandb_installed = False

from . import support_federated_generic
from . import support_federated_server

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class fed_avg_with_wandb_tracking(flwr.server.strategy.FedAvg):
    """
    A class that behaves like FedAvg but with the possibility to log the results through wandb.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for the training.
    all_config : dict
        Dictionayy with all the configuration parameters. It must contain the following keys:
        - server_config : dict
            Dictionary with the server configuration. 
        - dataset_config : dict
            Dictionary with the dataset configuration. The possible keys are the input parameters of the class inside src/dataset/dataset.py
        - model_config : dict
            Dictionary with the model configuration. The possible keys depends on the model used. See the files inside src/model for more information.
        - train_config : dict
            Dictionary with the training configuration. 
    *args, **kwargs : other parameters inherited from FedAvg
    """

    def __init__(self, model, all_config : dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Get config dictionaries
        server_config = all_config['server_config']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Wandb parameters
        
        if not wandb_installed : raise ImportError('wandb is not installed. Please it using "pip install wandb"')

        # Initialise wandb
        self.wandb_run = wandb.init(project = server_config['wandb_config']['project_name'], 
                                    job_type = "train", config = all_config, 
                                    notes = server_config['wandb_config']['notes'], 
                                    name = server_config['wandb_config']['name_training_run']
                                    )

        # Wandb artifact to save model weights
        self.model_artifact = wandb.Artifact(server_config['wandb_config']['model_artifact_name '], type = "model", 
                                             description = server_config['wandb_config']['description'] if 'description' in server_config['wandb_config'] else None,
                                             metadata = all_config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Other parameters

        self.count_rounds = 0
        self.tot_rounds = server_config['num_rounds']

        self.model = model

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Override methods from FedAvg

    def aggregate_fit(self, server_round: int, results, failures) :
        """
        Aggregate the results from the clients and upload the results in wandb
        """

        self.count_rounds += 1

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Parameters logging, clients metrics logging

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            model_weights = flwr.common.parameters_to_ndarrays(aggregated_parameters) 
            self.model_weights = model_weights

            # Load updated weights into the model
            support_federated_generic.set_weights(self.model, model_weights)

            # Save weights
            save_path = support_federated_generic.save_model_weights(self.model, path_to_save_model = self.server_config['path_to_save_model'], filename = f"model_round_{self.count_rounds}.pth")

            # Add weight to wandb
            self.model_artifact.add_file(save_path)
            wandb.save(save_path)
            
            # Iterate over clients
            for i in range(len(results)) :
                # Get metrics log dict for current client
                log_dict = results[i][1].metrics
                
                # Get id and number of training epoch
                client_id = log_dict['client_id']

                # Create epoch arrays
                training_epochs = np.arange(log_dict['epochs']) + 1

                # Extract losses
                metrics_values_list, metrics_name_list = support_federated_server.extract_metric_from_log_dict(log_dict)

                # Plot(s) creation and log
                # TODO Check which is better to log plot
                for metric_values, metric_name in zip(metrics_values_list, metrics_name_list) :
                    self.create_and_log_matplotlib_metric_plot([metric_values], training_epochs, [metric_name], client_id)
                    self.create_and_log_wandb_metric_plot([metric_values], training_epochs, [metric_name], client_id)
        else :
            model_weights = None

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Server metrics logging

        if aggregated_metrics is not None :
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
            self.wandb_run.log(wandb_log_dict)
        else :
            print("Warning. No aggregated metrics obtained during aggregation phase. fit_metrics_aggregation_fn must is None or there are error with the function code.")
            print("Only the aggregated weights and the metrics of the clients will be uploaded ")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        if self.count_rounds == self.tot_rounds :
            print("End training rounds")
            self.wandb_run.finish()

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate results from federated evaluation.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def evaluate(self, server_round, parameters):
        """
        Run centralized evaluation if callback was passed to strategy init.
        """
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # New function (i.e. all this funciton are not inherited from FedAvg)
    # TODO : consider if move them to support_federated_server

    def create_and_log_matplotlib_metric_plot(self, metrics_to_plot_list, training_epochs, metrics_name_list, client_id) :
        """
        Create a plot with matplotlib for the metric(s) inside metrics_to_plot_list and log it to wandb.
        """
        fig, _ = self.create_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id)
        
        # Name for the plot to log
        if len(metrics_name_list) == 1 :
            # Only 1 metric
            metric_name = metrics_name_list[0]
        else : 
            # More than 1 metric
            metric_name = 'all'
        
        # Log the plot in wandb
        self.wandb_run.log({
            f"{client_id}/{metric_name}_round_{self.count_rounds}_plt" : fig
        }, commit = False)

    def create_matplotlib_metric_plot(self, metrics_to_plot_list : list, training_epochs, metrics_name_list : list, client_id : str) :
        """
        Create a plot with matplotlib for the metrics in metrics_to_plot_list. Each element of the list must be an array/list with the value of the metric for each epoch.
        """

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

    def create_and_log_wandb_metric_plot(self, metrics_to_plot_list, training_epochs, metrics_name_list, client_id) :
        """
        Similar to create_and_log_matplotlib_metric_plot but the plot is created through wandb functions.
        Here more info https://docs.wandb.ai/guides/track/log/plots/ and https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA
        """
        fig, _ = self.create_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id)
        
        for i in range(len(metrics_to_plot_list)) :
            metric = metrics_to_plot_list[i]
            metric_name = metrics_name_list[i]

            # Convert the metric for wandb upload
            data = [[current_epoch, metric_value_current_epoch] for (current_epoch, metric_value_current_epoch) in zip(range(len(metric)), metric)]

            # Create the table 
            table = wandb.Table(data = data, columns=["x", "y"])

            # Log the plot in wandb
            self.wandb_run.log({
                f"{client_id}/{metric_name}_round_{self.count_rounds}" : wandb.plot.line(
                    table, "x", "y", title = f"{metric_name}"
                )
            }, commit = False)
            
            # Note on commit parameter
            # Description of commit parameter (from https://docs.wandb.ai/ref/python/log/)
            #   commit : If true, finalize and upload the step. If false, then accumulate data for the step. See the notes in the description. If step is None, then the default is commit=True; otherwise, the default is commit=False.
            # By default also step is None. So by default commit is True
            
            # Note on the / symbol inside the name of the object logged.
            # If I had the / symbol in the name all the log will be grouped together in a new section with that symbol.
            # E.g. if I log cleint_1/train_loss and client_1/test_loss I will create to plot called train_loss and test_loss in a specific NEW section called cleint_1
