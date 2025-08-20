"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import flwr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
                If True, the server will evaluate the model on the set saved in the server after each round. You must also specify the parameter evaluation_fn during the creation of the server.
                If False, the server will not evaluate the model on the server set after each round.
            - metrics_to_log_from_clients : list | str
                List of metrics to log from the clients. If None, no metrics will be logged. If 'all', all the metrics will be logged.
                If a list is provided the possible metrics are accuracy, choen_kappa, sensitivity, specificity, f1.
            - metrics_plot_backend : str
                The backend to use to create the clients metrics plot. This parameter is used only if the parameter metrics_to_log_from_clients is not None. It can be 'matplotlib' or 'wandb'.
        - dataset_config : dict
            Dictionary with the dataset configuration. The possible keys are the input parameters of the class inside src/dataset/dataset.py
        - model_config : dict
            Dictionary with the model configuration. The possible keys depends on the model used. See the files inside src/model for more information.
        - training_config : dict
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
        server_config   = all_config['server_config']
        wandb_config    = server_config['wandb_config']
        # training_config = all_config['training_config']

        self.all_config = all_config

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Wandb attributes
            
        self.wandb_run, self.model_artifact = support_federated_server.setup_wand(wandb_config, all_config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Other attributes

        self.count_rounds = 0                                             # Track the current number of round executed
        self.num_rounds           = server_config['num_rounds']           # Total number of federated round
        self.rounds_to_save_model = server_config['rounds_to_save_model'] # Save the model every n rounds
        self.metrics_to_log_from_clients = server_config['metrics_to_log_from_clients'] if 'metrics_to_log_from_clients' in server_config else None

        self.model = model

        wandb.watch(self.model, log = "all", log_freq = 1, log_graph = True)

    # def __del__(self):
    #     print("END SERVER AND WANDB LOG")
    #     self.wandb_run.finish()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Override methods from FedAvg

    def aggregate_fit(self, server_round: int, results, failures) :
        """
        Aggregate the results from the clients and upload the results in wandb
        """

        # IF executed here it throw an error
        # wandb.watch(self.model, log = "all", log_freq = 1, log_graph = True)

        # Call aggregate_fit from base class to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Log the data from client and servers
        aggregated_parameters, aggregated_metrics, self.model_weights, self.count_rounds = support_federated_server.aggregate_fit_log(aggregated_parameters, aggregated_metrics, server_round, results,
                                                                                                                                      self.model, self.wandb_run,
                                                                                                                                      self.count_rounds, self.rounds_to_save_model, self.num_rounds,
                                                                                                                                      self.all_config,
                                                                                                                                      self.model_artifact, self.metrics_to_log_from_clients
                                                                                                                                      )
        
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        """
        Run centralized evaluation if callback was passed to strategy init.
        """

        if self.all_config['server_config']['centralized_evaluation'] :
            # Compute metrics
            loss, test_metrics_dict = super().evaluate(server_round, parameters)
            
            # Log the metrics and the model
            support_federated_server.evaluate_log(test_metrics_dict, self.wandb_run, self.count_rounds, self.num_rounds, self.model_artifact)

            return loss, test_metrics_dict
        else :
            if self.count_rounds == 0 :
                print("")
                print("NO centralized evaluation function provided.")
