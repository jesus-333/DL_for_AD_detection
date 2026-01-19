
Second iteration of the flower app used to train the demnet. 

In the first iteration the data were read and split during the server instantiation (see function `prepare_data_for_FL_training` function inside the `server_app.py` file in the `ADNI_demnet_fedavg_with_wandb` folder).
This second iteration of the flower app will use instead the indices obtained from the `create_idx_files_for_federated_simulations.py` script.

