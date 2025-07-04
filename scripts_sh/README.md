Collection of shell scripts to train the models. Notes that most of the scripts are written with the purpose of running on an HPC with a SLURM scheduler (in my specific case the [HPC of unilu](https://hpc-docs.uni.lu/)).
The scripts with the suffix `_local` in the file name instead, could be executed on any machine (however all the scripts could be converted to local version removing the `srun` part from every command).

List of folders 
- `debug` : Contains script to debug and test the code.
- `train_demnet_FL` : Train the demnet model through Federated Learning
- `train_demnet_centralized` : Train the demnet model through classic centralized way
- `train_vgg_centralized` :  Train the VGG model through classic centralized way
- `train_vgg_centralized_parallel` : The purpose is the same of the scripts in `train_vgg_centralized`. The only difference is that this scripts launch multiple training of the same model at the same time (usefull if you have a GPU with a lot of VRAM). The scripts are separated from the `train_vgg_centralized` because the handling of the config toml file is a little different.
