# pretrain the darknet (first 20 conv layers in YOLO)
import torch
from darknet import DarkNet
import os
import torchvision.datasets as datasets 
import torchvision.transforms as T
from torch.utils.data import DataLoader, distributed
import matplotlib.pyplot as plt 
from PIL import Image
import time
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from hf_dataset import CustomImageNetDataset



log_dir = './log'
writer = SummaryWriter(log_dir=log_dir)  # TensorBoard log directory
num_epochs = 90
batch_size = 16
num_workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
best_accuracy = 0

def train_and_val(hf_dataset = "pkr7098/imagenet2012-1k-subsampling-50"):
    global best_accuracy 
    # print(torch.cuda.is_available())  # Should return True if CUDA is available
    # print(torch.cuda.current_device())  # Should show the current GPU device ID
    # print(torch.cuda.get_device_name(0))  # Should print the name of your GPU
    # print(device)
    # Dataset Loading
    print('loading dataset...')
    dataset = load_dataset(hf_dataset)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
                                T.RandomResizedCrop(224),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                normalize,])
    val_transform = T.Compose([
                              T.Resize(256),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              normalize,])
    train_dataset = CustomImageNetDataset(dataset['train'], transform=train_transform)
    val_dataset = CustomImageNetDataset(dataset['test'], transform=val_transform)

    model = DarkNet(batch_norm=True, conv_only=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                            momentum=momentum,
                            weight_decay=weight_decay)


    # Optimizer, criterion, etc.
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Data perparation
    print("loading train and val data set...")
    load_train_data_start = time.time()
    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size, 
                                shuffle = True, 
                                num_workers=num_workers, 
                                pin_memory=True
                                #sampler=train_sampler,
                            )

    load_train_data_end = time.time()
    print(f"finished loading train data set: {load_train_data_end - load_train_data_start:.3f}")

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    

    load_data_end = time.time()
    print(f"finished loading val data set: {load_data_end - load_train_data_end:.3f}")


    # Train
    for epoch in range(num_epochs):  # Define the number of epochs
      adjust_learning_rate(optimizer, epoch)
      model.train()  # Set the model to training mode for each epoch
      epoch_loss = 0.0  # Initialize the loss for the epoch
      total_train = 0
      correct_train = 0
      num_batches = len(train_loader)  # Get the number of batches
      # for batch_idx, (images, targets) in enumerate(train_loader):
      #   print(f"Batch {batch_idx}, Images shape: {images.shape}, Targets shape: {targets.shape}")
      #   print(f"Sample targets: {targets[:5]}")
      #   print(torch.unique(targets))
      with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear the gradients
            outputs = model(images)  # Forward pass through the model

            # print(f"Outputs shape: {outputs.shape}")  # Check the shape of outputs
            # print(f"Targets shape: {targets.shape}")  # Check the shape of targets

            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate the loss

            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            # Calculate and print the progress percentage
                # Update tqdm progress bar and display loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)  # Update the progress bar by 1 batch

        # Calculate train accuracy and loss
        avg_train_loss = epoch_loss / num_batches
        train_accuracy = (correct_train / total_train) * 100

        # Log training metrics to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
      # After each epoch, print summary
      print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss / num_batches:.4f}\n")

      # Validation
      model.eval()  # Set model to evaluation mode
      running_val_loss = 0.0
      val_correct = 0
      val_total = 0
      
      with torch.no_grad():  # Disable gradient calculations for validation
        for images, targets in val_loader:
          images, targets = images.to(device), targets.to(device)
          
          outputs = model(images)  # Forward pass
          loss = criterion(outputs, targets)  # Compute validation loss
          running_val_loss += loss.item()

          # Calculate accuracy
          _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
          val_total += targets.size(0)
          val_correct += (predicted == targets).sum().item()

      avg_val_loss = running_val_loss / len(val_loader)
      val_accuracy = (val_correct / val_total) * 100

      # Log validation metrics to TensorBoard
      writer.add_scalar('Val/Loss', avg_val_loss, epoch)
      writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
      print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n")

      is_best = val_accuracy > best_accuracy
      best_accuracy = max(val_accuracy, best_accuracy)
      state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            }
      save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth')
    
    return



    # plt.imshow(images_batch[0].permute(1, 2, 0)) # (C, H, W) -> (H, W, C) for visualization

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth'):
    path = os.path.join(log_dir, filename)
    torch.save(state, path)
    if is_best:
        path_best = os.path.join(log_dir, 'model_best.pth')
        shutil.copyfile(path, path_best)
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # TensorBoard
    writer.add_scalar('lr', lr, epoch)
    return 

    
if __name__ == '__main__':
    train_and_val()
    # list_plastic = os.listdir("/Users/ziqiaolin/Desktop/ML/YOLO/val/") 
    # number_files_plastic = len(list_plastic)
    # print(number_files_plastic)
    # Path to the folder containing images
    # folder_path = "/Users/ziqiaolin/Desktop/ML/YOLO/data/val/"

    # # List all files in the folder
    # file_list = os.listdir(folder_path)

    # # Filter out image files (jpg, png, etc.)
    # image_files = [f for f in file_list if f.endswith(('.png', '.jpg', '.JPEG', '.bmp', '.gif'))]
    # #print(image_files)
    # # Loop through image files and open each
    # for image_file in image_files[:2]:
    #     print(image_file)
    #     image_path = os.path.join(folder_path, image_file)
    #     img = Image.open(image_path)
    #     img.show()