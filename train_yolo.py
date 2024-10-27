import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import shutil
#from voc import VOCDataset
from darknet import DarkNet
from yolov1 import YOLOv1
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#assert device, 'Current implementation does not support CPU mode. Enable CUDA.'
print('CUDA current_device: {}'.format(torch.cuda.current_device()))
print('CUDA device_count: {}'.format(torch.cuda.device_count()))


# Path to data dir.
image_dir = 'data/VOC_allimgs/'

# Path to label files.
train_label = ('data/voc2007.txt', 'data/voc2012.txt')
val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
checkpoint_path = 'log/model_best.pth'

# Frequency to print/log the results.
print_freq = 5
tb_log_freq = 5

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64
log_dir = './yolo_log'

best_val_loss = float('inf')
writer = SummaryWriter(log_dir=log_dir) 
# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # Stack images
    targets = torch.stack([item[1] for item in batch], dim=0)  # Stack targets
    return images, targets

def train():
    global best_val_loss
    # Load pre-trained darknet.
    darknet = DarkNet(conv_only=True, batch_norm=True, init_weight=True)
    # darknet.features = torch.nn.DataParallel(darknet.features)  # not using right now

    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']

    # Load YOLO model.
    yolo = YOLOv1(darknet.features).to(device)
    yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layers)

    model_dict = yolo.state_dict()

    print("=> before loading ")
    check_conv_layers(yolo)
    # Filter the pretrained_dict to include only layers related to 'Conv2d' and 'BatchNorm2d'
    print('model dict weights')
    # for key,val in model_dict:
    #   print(key)
    # print(model_dict.keys())
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'features' in k}

    # Update the model's weights
    model_dict.update(pretrained_dict_filtered)
    yolo.load_state_dict(model_dict)

    print('darknet weight loading finished in yolo')
    #model_dict = darknet.state_dict()

    print("=> check yolo layers' weight ")
    check_conv_layers(yolo)

    # Setup loss and optimizer.
    criterion = Loss(feature_size=yolo.num_grids)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # data loader
    train_dataset = VOCDataset(is_train = True, normalize = True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=custom_collate_fn)

    val_dataset = VOCDataset(is_train = False, normalize = False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,drop_last=True,collate_fn=custom_collate_fn)

    # for data,target,label,_ in train_dataset:
    #   print(data.dtype, target.dtype)
      #print()

    # for data, targets in train_loader:
    #       print(f"Input type: {images.dtype}, shape: {images.shape}")
    #       print(f"Input type: {targets.dtype}, shape: {targets.shape}")  
        #print([target.shape for target in data])  # Print each target's shape

    # for images, targets,_,_ in train_loader:
    #     print(images.shape)  # Print image batch shape
    #     print([target.shape for target in targets])  # Print each target's shape
    #exit()
    # Train
    for epoch in range(config.WARMUP_EPOCHS + config.EPOCHS):  # Define the number of epochs

      yolo.train()  # Set the model to training mode for each epoch
      epoch_loss = 0.0  # Initialize the loss for the epoch
      total_train = 0
      correct_train = 0
      num_batches = len(train_loader)  # Get the number of batches


      with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (images,targets) in enumerate(train_loader):
            update_lr(optimizer, epoch, float(batch_idx) / float(len(train_loader) - 1))
            lr = get_lr(optimizer)
            # print(f"Input type: {images.dtype}, shape: {images.shape}")
            # print(f"Input type: {targets.dtype}, shape: {targets.shape}")            
            images, targets = images.to(device), targets.to(device)
            # print(f"Input type: {images.dtype}, shape: {images.shape}")
            # print(f"Input type: {targets.dtype}, shape: {targets.shape}")
            predicted = yolo(images)  # Forward pass through the model

            # print(f"Outputs shape: {outputs.shape}")  # Check the shape of outputs
            # print(f"Targets shape: {targets.shape}")  # Check the shape of targets

            loss = criterion(predicted, targets)  # Compute the loss
            optimizer.zero_grad()  # Clear the gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate the loss


            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            # Calculate and print the progress percentage
                # Update tqdm progress bar and display loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)  # Update the progress bar by 1 batch

        # Calculate train loss
        avg_train_loss = epoch_loss / num_batches

        # Log training metrics to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

      # After each epoch, print summary
      print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss / num_batches:.4f}\n")

      # Validation
      if epoch % 4 == 0:  # every 4 epoch will evaluate one time
        yolo.eval()       # Set model to evaluation mode
        running_val_loss = 0.0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient calculations for validation
          for images, targets in val_loader:
            images, targets = images.to(device).float(), targets.to(device).float()
            
            outputs = yolo(images)  # Forward pass
            loss = criterion(outputs, targets)  # Compute validation loss
            running_val_loss += loss.item()

            # Calculate loss
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
            val_total += targets.size(0)

        avg_val_loss = running_val_loss / len(val_loader)


        # Log validation metrics to TensorBoard
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)

        print(f"Validation Loss: {avg_val_loss:.4f}\n")

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss,best_val_loss)
        state = {
              'epoch': epoch + 1,
              'model_state_dict': yolo.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
              }
        save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth')

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth'):
    path = os.path.join(log_dir, filename)
    torch.save(state, path)
    if is_best:
        path_best = os.path.join(log_dir, 'model_best.pth')
        shutil.copyfile(path, path_best)
    return
if __name__ == '__main__':
  train()

