"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

from flwr.client import Client, NumPyClient
from flwr.common import Context

from .support_federated import get_weights, set_weights
from ..training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_weights(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_functions.train(self.net, self.trainloader, epochs=1)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = train_functions.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(context : Context) -> Client :
    """
    Create a Flower client representing a single organization.
    """

    # Load model
    # TODO
    model = None

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id = partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(model, trainloader, valloader).to_client()
