# <center>Pytorch Warmup Scheduler</center>

[![](https://img.shields.io/badge/developed-100%25-red.svg)]()
[![](https://img.shields.io/badge/license-MIT-black.svg)](https://choosealicense.com/licenses/mit/)
[![](https://img.shields.io/badge/version-v1.0.0-blue.svg)]()
[![](https://img.shields.io/badge/Author-LEFTeyes-pink.svg)](https://github.com/LEFTeyex)

## Description

A Warmup Scheduler for Pytorch to achieve the warmup learning rate at the beginning of training.

## setup

```
setup by pip will be achieved soon.
```

## Usage

Detail to see [run.py](run.py) file.

```python
import torch

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from warmup_module import WarmUpScheduler

model = Model()
optimizer = SGD(model.parameters(), lr=0.1)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
data_loader = torch.utils.data.DataLoader(...)
warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=len(data_loader),
                                   warmup_steps=100,
                                   warmup_start_lr=0.01,
                                   warmup_mode='linear')
epochs = 100
for epoch in range(epochs):
    for batch_data in data_loader:
        output = model(...)
        # loss = loss_fn(output, ...)
        # loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        warmup_scheduler.step()

    # lr_scheduler.step() is no longer needed
```