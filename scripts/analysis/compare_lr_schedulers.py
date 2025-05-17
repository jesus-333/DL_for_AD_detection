"""
This script is not related with the analysis of AD data or the trained model.
I simply use it to check how the learning rate (lr) change based on some schedulers
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

epochs = 200
starting_lr = 0.005

# Parameter of ExponentialLR
gamma = 0.95

# Parameters of CosineAnnealingLR
T_max = 10
eta_min = 1e-5

# Parameters of CosineAnnealingWarmRestarts
T_0 = T_max
T_mult = 2
eta_min = eta_min

# Parameters of CyclicLR
base_lr = 1e-5
max_lr = starting_lr
gamma = gamma
mode = 'exp_range'
step_size_up = 10
step_size_down = 15

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def simulate_lr_scheduler(lr_scheduler, optimizer, epochs):
    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr()[0])
    return lrs

model = torch.nn.Linear(10, 2)

# Simulate ExponentialLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
lrs_exp = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

# Simulate CosineAnnealingLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min = 0)
lrs_cos = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

# Simulate CosineAnnealingWarmRestarts
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_0, T_mult = T_mult, eta_min = 0)
lrs_cos_warm = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

# Simulate CyclicLR
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, gamma = gamma, step_size_up = step_size_up, step_size_down = step_size_down, mode = mode)
lrs_cyc = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

# Simulate ChainedScheduler
optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
schedulers_list = [
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min = 0),
]
lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers_list, optimizer)
lrs_chained_1 = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

optimizer = torch.optim.AdamW(model.parameters(), lr = starting_lr)
schedulers_list = [
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma),
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_0, T_mult = T_mult, eta_min = 0),
]
lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers_list, optimizer)
lrs_chained_2 = simulate_lr_scheduler(lr_scheduler, optimizer, epochs)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot the learning rates

epochs = range(1, epochs + 1)
fig, axs = plt.subplots(2, 3, figsize = (22, 8))
axs[0, 0].plot(epochs, lrs_exp, label = 'ExponentialLR')
axs[0, 0].set_title('ExponentialLR')
axs[0, 0].set_yscale('log')

axs[0, 1].plot(epochs, lrs_cos, label = 'CosineAnnealingLR')
axs[0, 1].set_title('CosineAnnealingLR')

axs[0, 2].plot(epochs, lrs_cos_warm, label = 'CosineAnnealingWarmRestarts')
axs[0, 2].set_title('CosineAnnealingWarmRestarts')

axs[1, 0].plot(epochs, lrs_cyc, label = 'CyclicLR')
axs[1, 0].set_title('CyclicLR')
axs[1, 0].set_yscale('log')

axs[1, 1].plot(epochs, lrs_chained_1, label = 'ChainedScheduler 1')
axs[1, 1].set_title('ChainedScheduler 1')
axs[1, 2].plot(epochs, lrs_chained_2, label = 'ChainedScheduler 2')
axs[1, 2].set_title('ChainedScheduler 2')

for ax in axs.flatten():
    ax.set(xlabel = 'Epochs', ylabel = 'Learning Rate')
    ax.set_xlim(1, epochs[-1])
    ax.grid()

fig.tight_layout()
plt.show()
