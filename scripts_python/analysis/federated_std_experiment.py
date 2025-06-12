"""
Simulate n clients with m samples per clients.
Compute the std of each client, the true std of all dataset together and the average of the std of the various clients.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

n_clients = 12
samples_per_client = 1000
sample_shape = [10, 10]  # Shape of each sample

scale = 5
loc = 2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_and_print_std(dataset, name: str) :
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Compute std

    # Entire dataset
    std_dataset = np.std(dataset)

    # Per client
    std_clients = np.std(dataset, axis = (1, 2, 3))

    # Average std of clients
    std_clients_avg = np.mean(std_clients)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Print
    
    print(f"Dataset: {name}")
    print(f"Std of the entire dataset : {std_dataset}")
    print(f"Average std of clients    : {std_clients_avg}")
    print(f"Std of each client        : {std_clients}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate data

dataset_normal      = np.random.normal(loc = loc, scale = scale, size = (n_clients, samples_per_client, *sample_shape))
dataset_uniform     = np.random.uniform(low = loc - scale, high = loc + scale, size = (n_clients, samples_per_client, *sample_shape))
dataset_exponential = np.random.exponential(scale = scale, size = (n_clients, samples_per_client, *sample_shape))
dataset_poisson     = np.random.poisson(lam = loc, size = (n_clients, samples_per_client, *sample_shape))
dataset_binomial    = np.random.binomial(n = 10, p = 0.5, size = (n_clients, samples_per_client, *sample_shape))

# Compute and print std for each dataset
compute_and_print_std(dataset_normal, "Normal Distribution")
compute_and_print_std(dataset_uniform, "Uniform Distribution")
compute_and_print_std(dataset_exponential, "Exponential Distribution")
compute_and_print_std(dataset_poisson, "Poisson Distribution")
compute_and_print_std(dataset_binomial, "Binomial Distribution")

