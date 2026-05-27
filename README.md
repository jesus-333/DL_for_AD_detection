# Deep Learning for AD Detection

This repository contains my work focused on the applications of Deep Learning (DL) for Alzheimer Disease (AD) detection.
This work is still in a very early stage. 

The repository is structured as a python package that can be built and installed through the [hatchling build system](https://hatch.pypa.io/latest/).

List of folders :
- `src` : Contains the actual source files that are built and installed by hatchling. So basically all the code for the models, training logic, dataset management is inside here.
- `scripts_python` : Python scripts with various purpose which are not part of the package but use its functions (e.g. training scripts, scripts to analyze or convert data etc)
- `scripts_sh` : Shell script to launch training of the models (OLD Version).
- `scripts_sh_V2` : Shell script to launch training of the models (New and improved version)
- `config` : Folders with config files (in [toml format](https://toml.io/en/)) used during the training. Note that this folder currently contains sample configuration files. The ones used during training can be updated and created via special scripts (and their location can be changed if necessary). Inside the `scripts_python` folder there are some scripts dedicated to create/update the config files.


## Install

### pip installation
Not yet implemented

### Build from source
Alternatively, you can download the repository and compile it locally via [hatchling](https://pypi.org/project/hatchling/)
```sh
pip install hatchling
git clone https://github.com/jesus-333/DL_for_AD_detection.git
cd DL_for_AD_detection 
hatchling build && pip install .
```

## CLI Commands

This package incudes also a list of CLI commands. Here's a short list. Each command has a `--help` flag with more details about how it works and all the options.

Data Manipulation
- `nii-to-hdf5` : Convert all the `nii` files inside a folder in a SINGLE `hdf5` file
- `npy-to-pt` : Convert all the `npy` files inside a folder (and eventual subfolder) into `pt` files.
- `pt-to-npy` : Opposite of `npy-to-pt`.

Wandb Interaction
- `wandb-get-run-status` : Given entity, project, and run id return the status of the current run (if exist) and print it in the terminal.

