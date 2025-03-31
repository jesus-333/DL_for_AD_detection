try :
    import flwr
except ImportError:
    raise ValueError("To use the federated learning module you need to install flwr. Run 'pip install flwr'.")

try :
    from flwr.simulation import run_simulation
except ImportError:
    print("Simulation module not detected in flower library. If you wish to run federated learning through simulation this module is required.")
    print("Please install the module by running ''pip install\"flwr[simulation]\".")
