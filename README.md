# <center>Warmup Scheduler Pytorch</center>

[![Tests](https://github.com/LEFTeyex/warmup/actions/workflows/tests.yaml/badge.svg)](https://github.com/LEFTeyex/warmup/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/LEFTeyex/warmup/branch/master/graph/badge.svg?token=E90TZPO40B)](https://codecov.io/gh/LEFTeyex/warmup)
[![PyPI version shields.io](https://img.shields.io/pypi/v/warmup-scheduler-pytorch.svg)](https://pypi.org/project/warmup-scheduler-pytorch/)
[![PyPI license](https://img.shields.io/pypi/l/warmup-scheduler-pytorch.svg)](https://pypi.org/project/warmup-scheduler-pytorch/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/warmup-scheduler-pytorch.svg)](https://pypi.python.org/pypi/warmup-scheduler-pytorch/)

## Description

A Warmup Scheduler for Pytorch to make the warmup learning rate change at the beginning of training.

## setup

Notice: need to install pytorch>=1.1.0 manually. \
The official website of pytorch is: https://pytorch.org/

Then install as follows:

```
pip install warmup_scheduler_pytorch
```

## Usage

Detail to see [example.py](example.py) file.

```python
import torch

from torch.optim import SGD  # example
from torch.optim.lr_scheduler import CosineAnnealingLR  # example

from warmup_scheduler_pytorch import WarmUpScheduler

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