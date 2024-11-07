import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import shutil
#from voc import VOCDataset
from models.yolov1net_resnet18 import YOLOv1_resnet18
from loss import Loss
from voc import VOCDataset
import os
import numpy as np
import math
from datetime import datetime
import config
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#assert device, 'Current implementation does not support CPU mode. Enable CUDA.'
# print('CUDA current_device: {}'.format(torch.cuda.current_device()))
# print('CUDA device_count: {}'.format(torch.cuda.device_count()))

# Path to data dir.
image_dir = 'data/VOC_allimgs/'

# Path to label files.
train_label = ('data/voc2007.txt', 'data/voc2012.txt')
val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
checkpoint_path = 'log/model_best.pth'

# Frequency to print/log the results.
tb_log_freq = 5

# Training hyper parameters.
init_lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4

log_dir = './yolo_log'
accum_iter = 16
best_val_loss = float('inf')
writer = SummaryWriter(log_dir=log_dir) 

def update_lr(optimizer, epoch):

    for g in optimizer.param_groups:
        # 1. linear increase from 0.00001 to 0.0001 over 5 epochs
        if epoch > 0 and epoch <= 5:
            g['lr'] = 0.00001 +(0.00009/5) * (epoch)
        # train at  0.0001 for 75 epochs 
        if epoch <=80 and epoch > 5:
            g['lr'] = 0.0001
        # train at 0.00001 for 30 epochs
        if epoch<= 110 and epoch> 80:
            g['lr'] = 0.00001
        # train until done
        if epoch > 110:
            g['lr'] = 0.000001

def check_conv_layers(model):
    for i, (name, layer) in enumerate(model.named_children()):
        # If the layer is a container (like Sequential), we need to recurse into it
        if isinstance(layer, (torch.nn.Sequential, torch.nn.Module)):
            check_conv_layers(layer)  # Recursively check the sublayers

        # Check if the layer is Conv2D
        if isinstance(layer, torch.nn.Conv2d):
            mean = layer.weight.mean().item()
            std = layer.weight.std().item()
            print(f"{name} - Layer {i} - Conv2D: Mean = {mean}, Std = {std}")
        
        # Check if the layer is BatchNorm2D
        if isinstance(layer, torch.nn.BatchNorm2d):
            mean = layer.weight.mean().item()
            std = layer.weight.std().item()  # Was `mean()` in your original code, should be `std()`
            print(f"{name} - Layer {i} - BatchNorm2D: Mean = {mean}, Std = {std}")
        
        if isinstance(layer, torch.nn.Linear):
            mean = layer.weight.mean().item()
            std = layer.weight.std().item()  # Was `mean()` in your original code, should be `std()`
            print(f"{name} - Layer {i} - Linear: Mean = {mean}, Std = {std}")
        
    return

def main():
    global best_val_loss

    train_transform = Compose([T.Resize((448, 448)),
                T.ColorJitter(brightness=[0,1.5], saturation=[0,1.5]),
                T.ToTensor()])

    test_transform = Compose([T.Resize((448, 448)),
                T.ToTensor()])
    #print(device)
    yolo = YOLOv1_resnet18().to(device)

    criterion = Loss(feature_size=yolo.num_grids)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # data loader
    train_dataset = VOCDataset(is_train = True, normalize = True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE//accum_iter, shuffle=True, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=utils.collate_function)
    train_loader_mAP = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=utils.collate_function)
    
    val_dataset = VOCDataset(is_train = False, normalize = False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE//accum_iter, shuffle=False, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=utils.collate_function)
    val_loader_mAP = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=utils.collate_function)

    # initialization
    train_loss_lst = []
    val_loss_lst = []
    train_mAP_lst = []
    val_mAP_lst = []

    # Train
    for epoch in range(config.WARMUP_EPOCHS + config.EPOCHS):  # Define the number of epochs
        
        # update learning rate
        update_lr(optimizer, epoch)

        # Train
        train_loss = train(train_loader, yolo, optimizer,  criterion, epoch)
        train_mAP = utils.evalute_map(train_loader_mAP, yolo, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
        train_loss_lst.append(train_loss)
        train_mAP_lst.append(train_mAP)
        print(f"Epoch:{epoch + 1 }  Train[Loss:{train_loss} mAP:{train_mAP}]")       

        # Validation
        if (epoch+1)%config.VAL_FREQ == 0:
            val_loss = test(val_loader, yolo, criterion)
            val_mAP = utils.evalute_map(val_loader_mAP, yolo, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
            val_loss_lst.append(val_loss)
            val_mAP_lst.append(val_mAP)
            print(f"Learning Rate:", optimizer.param_groups[0]["lr"])
            print(f"Epoch:{epoch + 1 }  Train[Loss:{train_loss} mAP:{train_mAP}]  Test[Loss:{val_loss} mAP:{val_mAP}]")
    

        # Log validation metrics to TensorBoard
        # writer.add_scalar('Val/Loss', avg_val_loss, epoch)

        # print(f"Validation Loss: {avg_val_loss:.4f}\n")

        # is_best = avg_val_loss < best_val_loss
        # best_val_loss = min(avg_val_loss,best_val_loss)
        # state = {
        #       'epoch': epoch + 1,
        #       'model_state_dict': yolo.state_dict(),
        #       'optimizer_state_dict': optimizer.state_dict()
        #       }
        # save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth')

def train(train_loader, model, optimizer, loss_f, epoch):
    """
    This function is to run the training process and calculate the loss for single epoch
    Input: 
        train loader (torch loader)
        model (torch model)
        optimizer (torch optimizer)
        loss function (torch custom yolov1 loss)
        epoch (int) current epoch number
    Output: loss (torch float).
    """
    model.train()  # Set the model to training mode for each epoch
    epoch_loss = 0.0  # Initialize the loss for the epoch
    total_train = 0
    correct_train = 0
    num_batches = len(train_loader)  # Get the number of batches
    
    with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", unit="batch") as pbar:
        for batch_idx, (data,targets, filename) in enumerate(train_loader):
            tgs = torch.cat([target['yolo_targets'].unsqueeze(0).float().to(device)for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in data], dim=0)
            
            with torch.set_grad_enabled(True):
                preds = model(im)
                loss = loss_f(preds, tgs)  # Compute the loss
                # optimizer.zero_grad()  # Clear the gradients
                epoch_loss += loss.item()
                loss.backward()  # Backpropagation

                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)  # Update the progress bar by 1 batch
        
        return epoch_loss/num_batches

def test(test_loader, model, loss_f):
    """
    This function is to run the testing process and calculate the loss for single epoch
    Input: test loader (torch loader), model (torch model), loss function 
          (torch custom yolov1 loss).
    Output: test loss (torch float).
    """
    epoch_loss = 0.0
    model.eval()
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (data,targets, filename) in enumerate(test_loader):
            tgs = torch.cat([target['yolo_targets'].unsqueeze(0).float().to(device)for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in data], dim=0)
            
            preds = model(im)
            loss = loss_f(preds, tgs)  # Compute the loss
            # optimizer.zero_grad()  # Clear the gradients
            epoch_loss += loss.item()

            del tgs, im, preds
    return epoch_loss/num_batches

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth'):
    path = os.path.join(log_dir, filename)
    torch.save(state, path)
    if is_best:
        path_best = os.path.join(log_dir, 'model_best.pth')
        shutil.copyfile(path, path_best)
    return
if __name__ == '__main__':
    main()

