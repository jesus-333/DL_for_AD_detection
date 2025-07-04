# Deep Learning for AD Detection

This repository contains my work focused on the applications of Deep Learning (DL) for Alzheimer Disease (AD) detection.
This work is still in a very early stage. 

The repository is structured as a python package that can be built and installed through the [hatchling build system](https://hatch.pypa.io/latest/).

List of folders :
- `src` : Contains the actual source files that are built and installed by hatchling. So basically all the code for the models, training logic, dataset management is inside here.
- `scripts_python` : Python scripts with various purpose which are not part of the package but use its functions (e.g. training scripts, scripts to analyze or convert data etc)
- `scripts_sh` : Shell script to launch training of the models
- `config` : Folders with config files (in [toml format](https://toml.io/en/)) used during the training. Note that this folder currently contains sample configuration files. The ones used during training can be updated and created via special scripts (and their location can be changed if necessary). Inside the `scripts_python` folder there are some scripts dedicated to create/update the config files.
