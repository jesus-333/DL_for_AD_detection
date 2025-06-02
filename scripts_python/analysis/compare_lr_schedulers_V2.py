"""
This script is not related with the analysis of AD data or the trained model.
I simply use it to check how the learning rate (lr) change based on some schedulers
This version create plot all the scheduler on the same plot
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

epochs = 60
starting_lr = 0.001

# Parameter of ExponentialLR
gamma_exp = 0.97

# Parameters of CosineAnnealingLR
T_max = 12
eta_min = 1e-4

# Parameters of CosineAnnealingWarmRestarts
T_0 = 12
T_mult = 1
eta_min = 1e-4

# Parameters of CyclicLR
base_lr = 1e-5
max_lr = 1e-3
gamma = 0.96
mode = 'exp_range'
# mode = 'triangular2'
step_size_up = 3
step_size_down = 10

figsize = (10, 6)
use_log_scale = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def simulate_lr_scheduler(lr_scheduler, optimizer, epochs):
    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr()[0])
    return lrs

model = torch.nn.Linear(10, 2)
list_lrs_to_plot = []
name_lrs_to_plot = []

# Simulate ExponentialLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma_exp)
lrs_exp = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
# list_lrs_to_plot.append(lrs_exp)
# name_lrs_to_plot.append('ExponentialLR')

# Simulate CosineAnnealingLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min = eta_min)
lrs_cos = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
list_lrs_to_plot.append(lrs_cos)
name_lrs_to_plot.append('CosineAnnealingLR')

# Simulate CosineAnnealingWarmRestarts
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_0, T_mult = T_mult, eta_min = eta_min)
lrs_cos_warm = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
# list_lrs_to_plot.append(lrs_cos_warm)
# name_lrs_to_plot.append('CosineAnnealingWarmRestarts')

# Simulate CyclicLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, gamma = gamma, step_size_up = step_size_up, step_size_down = step_size_down, mode = mode)
lrs_cyc = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
# list_lrs_to_plot.append(lrs_cyc)
# name_lrs_to_plot.append('CyclicLR')

# optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
# lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = 0.003, gamma = gamma, step_size_up = step_size_up, step_size_down = step_size_down, mode = mode)
# lrs_cyc = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
# list_lrs_to_plot.append(lrs_cyc)
# name_lrs_to_plot.append('CyclicLR 2')

# Simulate ChainedScheduler
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
schedulers_list = [
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min = eta_min),
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma_exp),
]
lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers_list, optimizer)
lrs_chained_1 = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)
list_lrs_to_plot.append(lrs_chained_1)
name_lrs_to_plot.append('ExponentialLR + CosineAnnealingLR')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot the learning rates

fig, ax = plt.subplots(figsize = figsize)

for i in range(len(list_lrs_to_plot)): ax.plot(list_lrs_to_plot[i], label = name_lrs_to_plot[i])

ax.set_xlabel('Epochs')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedulers')
if use_log_scale: ax.set_yscale('log')
ax.legend()
ax.grid()

fig.tight_layout()
plt.show()
