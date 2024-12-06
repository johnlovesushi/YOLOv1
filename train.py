import torch
from torch.utils.data import DataLoader
from loss import YoloV1Loss
from voc import VOCDataset
import config
import argparse
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import Subset
import torchvision.transforms as T
from utils import get_bboxes, mean_average_precision as mAP
from models.yolov1net_resnet18 import YOLOv1_resnet18
from models.yolov1net_vgg19bn import YOLOv1_vgg19bn
from models.yolov1net_resnet50 import YOLOv1_resnet50

init_lr = 0.00001
accum_iter = 16
weight_decay = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_dict = {'vgg19bn': lambda: YOLOv1_vgg19bn().to(device),
                   'resnet18bn': lambda: YOLOv1_resnet18().to(device),
                   'resnet50bn': lambda: YOLOv1_resnet50().to(device)
                   }

last_epoch = 0              # used for continue training


def parse():
    parser = argparse.ArgumentParser(description='PyTorch YOLOv1 Training')
    parser.add_argument('-m', "--model", dest='model_name',
                        type=str, default='resnet18bn')
    parser.add_argument('-r', '--resume', dest='resume_training',
                        action='store_true', help='resume the training process')
    parser.add_argument('-s', '--save_model', dest='save_model', action='store_false',
                        help='store the model state dict')
    parser.add_argument('--subset', dest='use_subset',
                        action='store_false', help='use subset to train the model')
    parser.add_argument('--subset_size', dest='subset_size',
                        default=200, help='the size of the subset for training')
    parser.add_argument('-d', '--destination',
                        dest='destination', type=str, default='results')
    args = parser.parse_args()
    print("args:", args)
    return args


def update_lr(optimizer, epoch):
    for g in optimizer.param_groups:
        # 1. linear increase from 0.00001 to 0.0001 over 5 epochs
        if epoch > 0 and epoch <= 5:
            g['lr'] = 0.00001 + (0.00009/5) * (epoch)
        # train at  0.0001 for 75 epochs
        if epoch <= 80 and epoch > 5:
            g['lr'] = 0.0001
        # train at 0.00001 for 30 epochs
        if epoch <= 110 and epoch > 80:
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
            # Was `mean()` in your original code, should be `std()`
            std = layer.weight.std().item()
            print(f"{name} - Layer {i} - BatchNorm2D: Mean = {mean}, Std = {std}")

        if isinstance(layer, torch.nn.Linear):
            mean = layer.weight.mean().item()
            # Was `mean()` in your original code, should be `std()`
            std = layer.weight.std().item()
            print(f"{name} - Layer {i} - Linear: Mean = {mean}, Std = {std}")

    return


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


def main():
    args = parse()
    # initialization
    if args.model_name in model_name_dict:
        # Call the corresponding function
        model = model_name_dict[args.model_name]()
        print(f"Initialized Model: {args.model_name}")
    else:
        raise ValueError(
            f"Invalid model name: {args.model_name}. Choose from {list(model_name_dict.keys())}")

    path_cpt_file = f'cpts/yolov1net_{args.model_name}.cpt'

    train_transform = Compose([T.Resize((448, 448)),
                               T.ColorJitter(
                                   brightness=[0, 1.5], saturation=[0, 1.5]),
                               T.ToTensor()])

    test_transform = Compose([T.Resize((448, 448)),
                              T.ToTensor()])

    criterion = YoloV1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # data loader
    train_dataset = VOCDataset(
        is_train=True, transform=train_transform, transform_scale_translate=True)
    val_dataset = VOCDataset(
        is_train=False, transform=test_transform, transform_scale_translate=True)

    if args.use_subset:
        print(f"using a subset to train the model\n")
        subset_indices = range(1, args.subset_size)
        train_dataset = Subset(train_dataset, subset_indices)
        val_dataset = Subset(val_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE //
                              accum_iter, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE//accum_iter,
                            shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True)

    # initialization
    train_loss_lst = []
    val_loss_lst = []
    train_mAP_lst = []
    val_mAP_lst = []

    # Train
    for epoch in range(config.WARMUP_EPOCHS + config.EPOCHS):  # Define the number of epochs

        # update learning rate
        update_lr(optimizer, epoch)
        train_pred_bbox, train_target_bbox = get_bboxes(train_loader, model, iou_threshold=0.5,
                                                        threshold=0.4)
        test_pred_bbox, test_target_bbox = get_bboxes(val_loader, model, iou_threshold=0.5,
                                                      threshold=0.4)
        # Train
        train_loss = train(train_loader, model, optimizer,  criterion, epoch)
        train_mAP = mAP(train_pred_bbox, train_target_bbox)
        train_loss_lst.append(train_loss)
        train_mAP_lst.append(train_mAP.item())

        # Val
        val_loss = test(val_loader, model, criterion)
        val_mAP = mAP(test_pred_bbox, test_target_bbox)
        val_loss_lst.append(val_loss)
        val_mAP_lst.append(val_mAP.item())
        print(
            f"Epoch:{epoch + 1}  Learning Rate:{optimizer.param_groups[0]["lr"]} Train[Loss:{train_loss} mAP:{train_mAP.item()}]  Test[Loss:{val_loss} mAP:{val_mAP.item()}]")

        if args.save_model and (((epoch + last_epoch + 1) % 2) == 0 or epoch + last_epoch == config.WARMUP_EPOCHS + config.EPOCHS - 1):
            state = {
                "epoch": epoch + last_epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, path_cpt_file)
            print(f"Checkpoint at {epoch + last_epoch + 1} stored")
            with open(f'{args.destination}/{args.model_name}train_loss.txt', 'w') as values:
                values.write(str(train_loss_lst))
            with open(f'{args.destination}/{args.model_name}train_mAP.txt', 'w') as values:
                values.write(str(train_mAP_lst))
            with open(f'{args.destination}/{args.model_name}test_loss.txt', 'w') as values:
                values.write(str(val_loss_lst))
            with open(f'{args.destination}/{args.model_name}test_mAP.txt', 'w') as values:
                values.write(str(val_mAP_lst))


def train(train_loader, model, optimizer, loss_f, epoch):
    """
    This function is to run the training process and calculate the loss for single epoch
    @params:
        train loader (torch loader)
        model (torch model)
        optimizer (torch optimizer)
        loss function (torch custom yolov1 loss)
        epoch (int) current epoch number
    @return:
        loss (torch float): cumulative epoch loss divided by the number of batches
    """
    model.train()  # Set the model to training mode for each epoch
    epoch_loss = 0.0  # Initialize the loss for the epoch
    num_batches = len(train_loader)  # Get the number of batches

    with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", unit="batch") as pbar:
        for batch_idx, (data, targets) in enumerate(train_loader):

            with torch.set_grad_enabled(True):
                preds = model(data)
                loss = loss_f(preds, targets)  # Compute the loss
                epoch_loss += loss.item()
                loss.backward()  # Backpropagation

                # not update graident for every accum_iter
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)  # Update the progress bar by 1 batch

        return epoch_loss/num_batches


def test(test_loader, model, loss_f):
    """
    This function is to run the testing process and calculate the loss for single epoch
    @params:
        test loader (torch loader),
        model (torch model),
        loss_f: loss function  (torch custom yolov1 loss).
    @return:
        test_loss (float): cumulative epoch loss divided by the number of batches
    """
    epoch_loss = 0.0
    model.eval()
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            preds = model(data)
            loss = loss_f(preds, targets)  # Compute the loss
            epoch_loss += loss.item()

            del data, targets, preds
    return epoch_loss/num_batches


if __name__ == '__main__':
    main()
