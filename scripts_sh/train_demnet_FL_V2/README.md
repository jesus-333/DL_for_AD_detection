Second iteration of the scripts used to launch the simulated FL training. 
The training works in the same way. What I change is how data are divided between train/validation and between the clients.

In the first iteration the data were read and split during the server instantiation (see function `prepare_data_for_FL_training` function inside the `server_app.py` files) 
This second iteration of the will use instead the indices obtained from the `create_idx_files_for_federated_simulations.py` script.
The python script will be launched before the flower app, directly from the sh script.
