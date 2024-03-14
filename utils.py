import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts
from IPython.display import clear_output
from matplotlib import colors, pyplot as plt
#%matplotlib inline
from tqdm.autonotebook import tqdm, trange
from tqdm import tqdm, tqdm_notebook
from itertools import chain

def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))     
    mean = np.array([0.485, 0.456, 0.406])     
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)                   
    plt_ax.imshow(inp)
    plt_ax.grid(False)


# включим подсчет градиентов для слоев:
def parameters_grad(model):
    for name, p in model.named_parameters():
        p.requires_grad = True
    return model

# Определим процессор, на котором будут выполняться вычисления
def set_device():
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    print('was launched in:' ,device)
    return device


def set_optim_sched(model, baseline = False):
    if baseline:
        optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=3e-4,
                    betas=(0.9, 0.999),
                    weight_decay=0.01)

        scheduler = CyclicLR(optimizer, 
                     base_lr = 1e-6, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                     max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
                     step_size_up = 4, # Number of training iterations in the increasing half of a cycle
                     mode = "triangular2", 
                     cycle_momentum=False)

    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01)

    # Выберем CosineAnnealingWarmRestarts, такой чтобы шаг обучения периодически использовался вновь (график ниже)
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0 = 8,# Number of iterations for the first restart
                                                T_mult = 1, # A factor increases TiTi​ after a restart
                                                eta_min = 5e-6)
    return optimizer, scheduler

def save_checkpoint(state, filename="/kaggle/working/celeba_500_class/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def plot_training(train_losses, valid_losses):
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(train_losses, label='train_loss')
    plt.plot(valid_losses, label='valid_loss')
    plt.legend()
def plot_acc(train_accs, valid_accs):
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(train_accs, label='train_acc')
    plt.plot(valid_accs, label='valid_acc')
    plt.legend()
def plot_lr(lrates):
    plt.figure(figsize=(12, 3))
    #plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(lrates, label='lrates', marker=11)
    plt.legend()