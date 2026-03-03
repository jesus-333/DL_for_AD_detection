Train the VGG model in a centralized way.

The scripts with the `_parallel` suffix were created to test the feasibility to launch multiple training runs on the same GPU (useful if you have a GPU with a lot of VRAM). 
Note that the way these scripts handle of the config toml file is a little different from the non-parallel scripts.
