"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import flwr
import matplotlib
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
            Dictionary with the server configuration. Note that not all the keys are used inside this class. Some of them are used in the server_app creation (see for example the file server_app.py in scripts/training_FL/fedavg_with_wandb).
            The possible keys are:
            - num_rounds : int
                Number of rounds to run.
            - n_client : int
                Number of clients to use. It must be equal to the number of supernodes specified in pyproject.toml.
                This parameter is not used inside the class but it is used in the server_app.py file.
            - fraction_fit : float
                Fraction of clients to use for training. It must be between 0 and 1.
                This parameter is not used inside the class but it is used in the server_app.py file.
            - fraction_evaluate : float
                Fraction of clients to use for evaluation. It must be between 0 and 1.
                This parameter is not used inside the class but it is used in the server_app.py file.
            - keep_labels_proportion : bool
                If True when the data are divided among the clients, the labels proportions are kept (e.g. if the original data has 10% of class A and 90% of class B, the data for each client will have the same proportion). 
                If False, the data are divided uniformly among the clients without considering the labels proportions.
                This parameter is not used inside the class but it is used in the server_app.py file.
                Note that this is used for FL simulation.
            - centralized_evaluation : bool
                If True, the server will evaluate the model on the validation set after each round. You must also specify the parameter evaluation_fn during the creation of the server.
                If False, the server will not evaluate the model on the validation set after each round.
            - metrics_to_log_from_clients : list | str
                List of metrics to log from the clients. If None, no metrics will be logged. If 'all', all the metrics will be logged.
                If a list is provided the possible metrics are accuracy, choen_kappa, sensitivity, specificity, f1.
            - metrics_plot_backend : str
                The backend to use to create the clients metrics plot. This parameter is used only if the parameter metrics_to_log_from_clients is not None. It can be 'matplotlib' or 'wandb'. 
        - dataset_config : dict
            Dictionary with the dataset configuration. The possible keys are the input parameters of the class inside src/dataset/dataset.py
        - model_config : dict
            Dictionary with the model configuration. The possible keys depends on the model used. See the files inside src/model for more information.
        - train_config : dict
            Dictionary with the training configuration. See the documentation of train() function inside src/train_functions.py for more information about possible keys.
    *args, **kwargs : other parameters inherited from FedAvg

    Attributes
    ----------
    all_config : dict
        Dictionary with all the configuration parameters. See above for more information. This dictionary will be logged on wandb as run config and as artifact metadata.
    wandb_run : wandb.run
        Wandb run object. It is used to log the results on wandb. See https://docs.wandb.ai/ref/python/init and https://docs.wandb.ai/ref/python/run/ for more information.
    model_artifact : wandb.Artifact
        Wandb artifact object. It is used to save the model weights on wandb. See https://docs.wandb.ai/ref/python/artifact for more information.
    count_rounds : int
        Counter for the number of rounds. It is used as step for wandb logging.
    num_rounds : int
        Number of rounds to run. It is used to stop the wandb run after the last round. The value is taken from the server_config dictionary.
    metrics_to_log_from_clients : list | str
        List of metrics to log from the clients. If None, no metrics will be logged. If 'all', all the metrics will be logged.
        If a list is provided the possible metrics are accuracy, choen_kappa, sensitivity, specificity, f1.
        It is taken from the server_config dictionary.
    model : torch.nn.Module
        The model to use for the training. Its weights will be saved in wandb as pth file.
    """

    def __init__(self, model, all_config : dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Get config dictionaries
        server_config = all_config['server_config']
        training_config  = all_config['training_config']

        self.all_config = all_config

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Wandb attributes
        
        if not wandb_installed : raise ImportError('wandb is not installed. Please it using "pip install wandb"')

        # Initialise wandb
        self.wandb_run = wandb.init(project = training_config['project_name'], 
                                    job_type = "train", config = all_config, 
                                    notes = training_config['notes'], 
                                    name = training_config['name_training_run']
                                    )

        # Wandb artifact to save model weights
        self.model_artifact = wandb.Artifact(training_config['model_artifact_name'], type = "model", 
                                             description = training_config['description'] if 'description' in training_config else None,
                                             metadata = all_config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Other attributes

        self.count_rounds = 0
        self.num_rounds = server_config['num_rounds']
        self.metrics_to_log_from_clients = server_config['metrics_to_log_from_clients'] if 'metrics_to_log_from_clients' in server_config else None

        self.model = model

    def __del__(self):
        print("END SERVER AND WANDB LOG")
        self.wandb_run.finish()

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
            save_path = support_federated_generic.save_model_weights(self.model, path_to_save_model = self.all_config['training_config']['path_to_save_model'], filename = f"model_round_{self.count_rounds}.pth")

            # Add weight to wandb
            self.model_artifact.add_file(save_path)
            wandb.save(save_path)

            if self.metrics_to_log_from_clients is not None :
            
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
                    if self.metrics_to_log_from_clients == 'all' :
                        self.create_and_log_wandb_metric_plot_separately(metrics_values_list, training_epochs,  metrics_name_list, client_id)
                    else :
                        # Get the metrics to plot
                        idx_of_metrics_to_plot = [i for i in range(len(metrics_values_list)) if metrics_name_list[i] in self.metrics_to_log_from_clients]
                        metrics_values_to_plot_list = [metrics_values_list[idx] for idx in idx_of_metrics_to_plot]
                        metrics_name_to_plot_list   = [metrics_name_list[idx] for idx in idx_of_metrics_to_plot]
                        
                        # TODO add option to decide if upload also the plot with the metrics merged
                        self.create_and_log_wandb_metric_plot_separately(metrics_values_to_plot_list, training_epochs, metrics_name_to_plot_list, client_id)

                        # For now this not produce the plot I want
                        # self.create_and_log_wandb_metric_plot_together(metrics_values_to_plot_list, training_epochs, metrics_name_to_plot_list, client_id)

                    # for metric_values, metric_name in zip(metrics_values_list, metrics_name_list) :
                    #     self.create_and_log_matplotlib_metric_plot([metric_values], training_epochs, [metric_name], client_id)
                    #     self.create_and_log_wandb_metric_plot_separately([metric_values], training_epochs, [metric_name], client_id)
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
            print("FIT ", self.count_rounds)
            self.wandb_run.log(wandb_log_dict, step = self.count_rounds)
        else :
            print("Warning. No aggregated metrics obtained during aggregation phase. fit_metrics_aggregation_fn must is None or there are error with the function code.")
            print("Only the aggregated weights and the metrics of the clients will be uploaded ")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        if self.count_rounds == self.num_rounds and not self.all_config['server_config']['centralized_evaluation'] :
            self.end_wandb_run_and_log_artifact()
            print("End training rounds")

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        """
        Run centralized evaluation if callback was passed to strategy init.
        """

        if self.all_config['server_config']['centralized_evaluation'] :
            # Compute metrics
            loss, test_metrics_dict = super().evaluate(server_round, parameters)
        
            # Upload metrics
            self.wandb_run.log(test_metrics_dict, step = self.count_rounds)

            # Close wandb run if I'm at the last round
            if self.count_rounds == self.num_rounds and not self.all_config['server_config']['centralized_evaluation'] :
                self.end_wandb_run_and_log_artifact()
                print("End training rounds")

            # Store and log
            return loss, test_metrics_dict 


        else :
            if self.count_rounds == 0 :
                print("")
                print("NO centralized evaluation funciton provided.")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # New function (i.e. all this funciton are not inherited from FedAvg)
    # TODO : consider if move them to support_federated_server

    def end_wandb_run_and_log_artifact(self) :
        """
        Load the artifacts in wandb and finish the run.
        """
        self.wandb_run.log_artifact(self.model_artifact)
        self.wandb_run.finish()

    def create_matplotlib_metric_plot(self, metrics_to_plot_list : list, training_epochs, metrics_name_list : list, client_id : str) :
        """
        Create a plot with matplotlib for the metrics in metrics_to_plot_list. Each element of the list must be an array/list with the value of the metric for each epoch.
        """
        
        # Apparently matplotlib is not thread safe. 
        # The creation of fig and ax with the command plt.subplots() without using the default backend will cause the following warning and the crash of python.
        # UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
        # A possibile solution suggested online was the use of a non-interactive backend
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

    def create_and_log_matplotlib_metric_plot(self, metrics_to_plot_list, training_epochs, metrics_name_list, client_id) :
        """
        Create a plot with matplotlib for the metric(s) inside metrics_to_plot_list and log it to wandb.
        """
        fig, _ = self.create_matplotlib_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id)
        
        # Name for the plot to log
        if len(metrics_name_list) == 1 :
            # Only 1 metric
            metric_name = metrics_name_list[0]
        else : 
            # More than 1 metric
            metric_name = 'all'
        
        # Log the plot in wandb
        self.wandb_run.log({
            f"client_{client_id}/{metric_name}_round_{self.count_rounds}_plt" : fig
        }, commit = False)

        plt.close()

    def create_and_log_wandb_metric_plot_separately(self, metrics_to_plot_list, training_epochs, metrics_name_list, client_id) :
        """
        Similar to create_and_log_matplotlib_metric_plot but the plot is created through wandb functions.
        Note that in this function create a single plot for every metric you want to log. If you want to log more metrics in a single plot use create_and_log_wandb_metric_plot_together
        Here more info https://docs.wandb.ai/guides/track/log/plots/ and https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA
        """
        
        for i in range(len(metrics_to_plot_list)) :
            metric = metrics_to_plot_list[i]
            metric_name = metrics_name_list[i]

            # Convert the metric for wandb upload
            data = [[current_epoch, metric_value_current_epoch] for (current_epoch, metric_value_current_epoch) in zip(range(len(metric)), metric)]

            # Create the table 
            table = wandb.Table(data = data, columns=["epochs", metric_name])

            # Log the plot in wandb
            self.wandb_run.log({
                f"client_{client_id}/{metric_name}_round_{self.count_rounds}" : wandb.plot.line(
                    table, "epochs", metric_name, title = f"{metric_name}_round_{self.count_rounds}"
                )
            }, commit = False)
            
            # Note on commit parameter
            # Description of commit parameter (from https://docs.wandb.ai/ref/python/log/)
            #   commit : If true, finalize and upload the step. If false, then accumulate data for the step. See the notes in the description. If step is None, then the default is commit=True; otherwise, the default is commit=False.
            # By default also step is None. So by default commit is True
            
            # Note on the / symbol inside the name of the object logged.
            # If I had the / symbol in the name all the log will be grouped together in a new section with that symbol.
            # E.g. if I log cleint_1/train_loss and client_1/test_loss I will create to plot called train_loss and test_loss in a specific NEW section called cleint_1


    def create_and_log_wandb_metric_plot_together(self, metrics_to_plot_list, training_epochs, metrics_name_list, client_id) :
        """
        Similar to create_and_log_matplotlib_metric_plot_separately but the metrics are plotted in the same figure.
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
        
        plot_log_name += f"round_{self.count_rounds}"

        # Create the table 
        table = wandb.Table(data = data, columns = ["epochs", *metrics_name_list])

        # Log the plot in wandb
        self.wandb_run.log({
            plot_log_name : wandb.plot.line(
                table, "epochs", "Metrics value", title = f"{[metrics_name_list]}_round_{self.count_rounds}"
            )
        }, commit = False)
