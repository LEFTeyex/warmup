import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.warmup_scheduler_pytorch.warmup_module import WarmUpScheduler


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 1, (1, 1))

    def forward(self, x):
        return self.conv(x)


def get_lr(optimizer):
    return [p['lr'] for p in optimizer.param_groups][0]


def run():
    model = Model()
    optimizer = SGD(model.parameters(), lr=0.1)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
    len_dataloader = 50  # assume the len of dataloader is 50
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                       len_loader=len_dataloader,
                                       warmup_steps=100,
                                       warmup_start_lr=0.01,
                                       warmup_mode='linear')
    # training
    epochs = 100
    epoch_lr = [[], []]  # epoch, lr

    for epoch in range(epochs):
        for step in range(len_dataloader):

            if not warmup_scheduler.warmup_done:
                epoch_lr[0].append(epoch + step / len_dataloader)
                epoch_lr[1].append(get_lr(optimizer))
            else:
                epoch_lr[0].append(epoch)
                epoch_lr[1].append(get_lr(optimizer))

            # output = model(...)
            # loss = loss_fn(output, label)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            warmup_scheduler.step()

    plt.plot(*epoch_lr)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.show()


if __name__ == '__main__':
    run()
