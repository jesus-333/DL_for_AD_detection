"""
With this script you can fine tune the vgg neural network to classify MRI and fMRI data for alzheimer detection

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import toml

from src.model import vgg_nets
import src.training as training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_config = 'training_scripts/config/vgg_finetuning.toml'

finetuning_type = 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load config

all_config = toml.load(path_config)

train_config = all_config['train_config']
model_config = all_config['model_config']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load model
model, preprocess_functions = vgg_nets.get_vgg(model_config)

# Set type of finetuning
model.set_model_for_finetuning(finetuning_type)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

