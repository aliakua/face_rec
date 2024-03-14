from dataset import CelebaDataset 
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.autonotebook import tqdm, trange
from tqdm import tqdm, tqdm_notebook
from itertools import chain

from model import Backbone_Resnet, Baseline_Net, Angular_clf, Angular_Net
from loss import Angular_Loss
from utils import set_device, set_optim_sched, parameters_grad, save_checkpoint, plot_training, plot_acc, plot_lr


def train(train_loader, val_loader, model, criterion, epochs, LOAD_MODEL_FILE, tol = 5e-4):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    #SETTING
    device = set_device()
    optimizer , scheduler = set_optim_sched(model)
    model = model.to(device)
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
    best_model_wts = {}
    lrs = []
    best_acc = 0.0

    since = time.time()
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss: {v_loss:0.4f} train_acc: {t_acc:0.4f} val_acc: {v_acc:0.4f}"

    for epoch in range(epochs):
        torch.cuda.memory.empty_cache()
        print('epoch -', epoch+1)
        lrs.append(optimizer.param_groups[0]['lr'])
        print('learning rate', lrs[-1])

        # fit
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0
        batches = 0

        for inputs, labels in tqdm(train_loader):
            batches+=1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, 1)
            running_loss += loss #.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_size += inputs.size(0)
        train_loss = running_loss.cpu().detach().numpy() / batches
        train_acc = running_corrects/ processed_size
        # сделаем шаг обучения
        scheduler.step()

        model.eval()
        valid_loss = 0.0
        valid_corrects = 0
        valid_processed_size = 0
        valid_batches = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                valid_batches+=1
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, 1)

                valid_loss += loss#.item() * inputs.size(0)
                valid_corrects += torch.sum(preds == labels.data)
                valid_processed_size += inputs.size(0)
            val_loss = valid_loss.cpu().detach().numpy() / valid_batches
            val_acc = valid_corrects / valid_processed_size

                # save
            train_losses.append(train_loss)
            valid_losses.append(val_loss)
            train_accuracies.append(train_acc.cpu().detach().numpy())
            valid_accuracies.append(val_acc.cpu().detach().numpy())

        # если достиглось лучшее качество, то запомним веса модели
        if val_acc > best_acc and val_acc>0.4:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        

        tqdm.write(log_template.format(ep=epoch+1, t_loss=train_losses[-1],\
                                       v_loss=valid_losses[-1], t_acc=train_accuracies[-1], v_acc=valid_accuracies[-1]))

        if epoch>3:
            if abs(valid_accuracies[-2] - valid_accuracies[-1] ) < tol and abs(train_accuracies[-2] - train_accuracies[-1] ) < tol :
                print(f"\nConvergence. Stopping iterations.")
                #stop_it = True
                break

    model.load_state_dict(best_model_wts) # загрузим лучшие веса модели
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, lrs, train_losses, valid_losses, train_accuracies, valid_accuracies

def predict(model, test_loader, device):
    processed_size_ts =0
    running_corrects_ts =0
    test_accuracies = []

    with torch.no_grad():
        logits = []
        model.eval()
        for inputs, labels  in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, 1)
            running_corrects_ts += torch.sum(preds == labels.data).item()
            processed_size_ts += inputs.size(0)
        test_acc = running_corrects_ts / processed_size_ts

    return test_acc

def perform(model, lrates , train_losses, valid_losses , train_accs, valid_accs, ts_loader):
    plot_training(train_losses, valid_losses)
    plot_acc(train_accs, valid_accs)
    plot_lr(lrates)
    test_acc = predict(model,  ts_loader, device = set_device())
    print('Accurace on test set:', test_acc)

def main():
    # разные режимы датасета
    DATA_MODES = ['train', 'valid', 'test']
    # все изображения будут масштабированы к размеру 160*160 px
    RESCALE_SIZE = 160

    augmentation_pipeline_1 = A.Compose(
                                    [A.ColorJitter( brightness=0.1, contrast=0.6, saturation=0.2, hue=0.1),
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                                    ToTensorV2() # convert the image to PyTorch tensor
        ])

    train_dataset = CelebaDataset(mode='train')
    valid_dataset = CelebaDataset(mode='valid')
    test_dataset = CelebaDataset(mode='test')

    train_dataset_aug_1 = CelebaDataset(mode='train', augmentations = augmentation_pipeline_1)
    train = train_dataset  + train_dataset_aug_1 # new train consists of train and augmented datasets

    dataloader = {
            'train': torch.utils.data.DataLoader(
                dataset = train,
                batch_size = 32, # батч из 32 картинок
                shuffle = True,
                num_workers = 2),
            'val': torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size =32,
                shuffle = False,
                num_workers = 2),
            'ts': torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size =16,
                shuffle = False,
                num_workers = 2),
    }

    dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'val','ts']}
    dataset_sizes

    emb_size = 256
    out_num = 500 # 500 classes

    backbone = Backbone_Resnet()
    Softmax_CE = Baseline_Net(backbone, emb_size, out_num)
    Softmax_Angular = Angular_clf(emb_size, out_num)

    models = {'baseline': Softmax_CE,
            'arcface': Angular_Net(backbone, Softmax_Angular, emb_size),
            'sphereface': Angular_Net(backbone, Softmax_Angular, emb_size),
            'cosface': Angular_Net(backbone, Softmax_Angular, emb_size)}
    criterions = {'Softmax_CE': nn.CrossEntropyLoss(),
                'Softmax_Angular_arc': Angular_Loss('arcface', out_num),
                'Softmax_Angular_sphere': Angular_Loss('sphereface', out_num),
                'Softmax_Angular_cos': Angular_Loss('cosface', out_num)}
    #baseline
    model_baseline = parameters_grad(models ['baseline'])
    LOAD_MODEL_FILE = '/kaggle/working/celeba_500_class/baseline_overfit.pth.tar'
    model, lrates , train_losses, valid_losses , train_accs, valid_accs = train(dataloader['train'], 
                                                                                dataloader['val'], 
                                                                                model_baseline, 
                                                                                criterions['Softmax_CE'], 
                                                                                epochs=40)
    perform(model_baseline, lrates , train_losses, valid_losses , train_accs, valid_accs, dataloader['ts'])
    
    #arcface
    #model_arcface = models ['arcface']
    model_arcface = parameters_grad(models ['arcface'])
    LOAD_MODEL_FILE = '/kaggle/working/celeba_500_class/arcface_overfit.pth.tar'
    model_arcface, lrates , train_losses, valid_losses , train_accs, valid_accs = train(dataloader['train'], 
                                                                                        dataloader['val'], 
                                                                                        model_arcface, 
                                                                                        criterions['Softmax_Angular_arc'], 
                                                                                        epochs=60)
    perform(model_arcface, lrates , train_losses, valid_losses , train_accs, valid_accs, dataloader['ts'])                            
    
    #sphereface
    model_sphereface = parameters_grad(models ['sphereface'])
    LOAD_MODEL_FILE = '/kaggle/working/celeba_500_class/sphereface_overfit.pth.tar'
    model_sphereface, lrates , train_losses, valid_losses , train_accs, valid_accs = train(dataloader['train'], 
                                                                                           dataloader['val'], 
                                                                                           model_sphereface, 
                                                                                           criterions['Softmax_Angular_sphere'], 
                                                                                           epochs=40)
    perform(model_sphereface, lrates , train_losses, valid_losses , train_accs, valid_accs, dataloader['ts']) 

    # cosface
    model_cosface = parameters_grad(models ['cosface'])
    LOAD_MODEL_FILE = '/kaggle/working/celeba_500_class/cosface_overfit.pth.tar'
    model_cosface, lrates , train_losses, valid_losses , train_accs, valid_accs = train(dataloader['train'], 
                                                                                        dataloader['val'], 
                                                                                        model_cosface, 
                                                                                        criterions['Softmax_Angular_cos'], 
                                                                                        epochs=40)
    perform(model_cosface, lrates , train_losses, valid_losses , train_accs, valid_accs, dataloader['ts']) 


if __name__ == "__main__":
    #train
    main()