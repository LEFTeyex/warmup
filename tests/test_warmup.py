import torch.nn as nn

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.warmup_scheduler_pytorch.warmup_module import WarmUpScheduler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 1, (1, 1))

    def forward(self, x):
        return self.conv(x)


class TestWarmupScheduler(object):
    def setup_method(self):
        self.epochs = 100
        self.len_dataloader = 50

        self.model = Model()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.01)
        self.warmup_scheduler = WarmUpScheduler(self.optimizer, self.lr_scheduler,
                                                len_loader=self.len_dataloader,
                                                warmup_steps=2 * self.len_dataloader,
                                                warmup_start_lr=0.01,
                                                warmup_mode='linear', verbose=True)

    def test_warmup_step(self):
        for epoch in range(self.epochs):
            for step in range(self.len_dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.warmup_scheduler.step()

        self.warmup_scheduler.step(step=100)
        self.warmup_scheduler.step(epoch=100)
        self.warmup_scheduler.step(step=100, epoch=100)

    def test_warmup_init(self):
        optimizer = None
        lr_scheduler = None
        warmup_scheduler = WarmUpScheduler(self.optimizer, self.lr_scheduler,
                                           warmup_steps=1,
                                           warmup_start_lr=[0.01])

        try:
            warmup_scheduler = WarmUpScheduler(optimizer, self.lr_scheduler,
                                               warmup_steps=1,
                                               warmup_start_lr=0.01)
        except TypeError:
            pass

        try:
            warmup_scheduler = WarmUpScheduler(self.optimizer, lr_scheduler,
                                               warmup_steps=1,
                                               warmup_start_lr=0.01)
        except TypeError:
            pass

        try:
            optimizer = SGD(self.model.parameters(), lr=0.1)
            warmup_scheduler = WarmUpScheduler(optimizer, self.lr_scheduler,
                                               warmup_steps=1,
                                               warmup_start_lr=0.01)
        except KeyError:
            pass

        try:
            warmup_scheduler = WarmUpScheduler(self.optimizer, self.lr_scheduler,
                                               warmup_steps=1,
                                               warmup_start_lr=[0.01, 0.01])
        except AssertionError:
            pass

    def test_warmup_state_dict(self):
        sd = self.warmup_scheduler.state_dict()
        self.warmup_scheduler.load_state_dict(sd)

    def test_warmup_get(self):
        self.warmup_scheduler.get_last_lr()
        self.warmup_scheduler.get_warmup_lr()

        try:
            warmup_scheduler = WarmUpScheduler(self.optimizer, self.lr_scheduler,
                                               len_loader=self.len_dataloader,
                                               warmup_steps=2 * self.len_dataloader,
                                               warmup_start_lr=0.01,
                                               warmup_mode='rect')
        except ValueError:
            pass

    def test_warmup_done(self):
        done = self.warmup_scheduler.warmup_done
        assert isinstance(done, bool)
