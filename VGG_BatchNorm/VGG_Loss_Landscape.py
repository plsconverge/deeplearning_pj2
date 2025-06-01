# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
# from IPython import display
from torchvision import transforms, datasets

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
# from data.loaders import get_cifar_loader


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
def get_cifar_loader(train_flag, batch_size=128, num_workers=4, shuffle=True):
    # no data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=train_flag, download=False, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

# for X,y in train_loader:
#     ## --------------------
#     # Add code as needed
#     ## --------------------
#     break



# This function is used to calculate the accuracy of model classification
def get_accuracy(loader, model, device):
    ## --------------------
    # Add code as needed
    model.eval()
    total = 0
    correct = 0
    for image, labels in loader:
        image, labels = image.to(device), labels.to(device)
        logit = model.forward(image)
        _, prediction = torch.max(logit, 1)
        correct += torch.sum(torch.eq(prediction, labels)).item()
        total += labels.size(0)
    accuracy = correct / total
    return accuracy
    ## --------------------

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, warm=5, cycle=95, last_epoch=-1):
        self.warm, self.cycle = warm, cycle
        self.lr = init_lr
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def step(self):
        self.last_epoch += 1
        epoch = self.last_epoch

        if epoch < self.warm:
            curr_lr = self.lr * (epoch + 1) / self.warm
        else:
            curr_lr = self.lr * 0.5 * (1 + torch.cos(torch.tensor((epoch - 5) * torch.pi / self.cycle)))

        for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, device=torch.device('cpu')):
    model.to(device)
    # learning_curve = [np.nan] * epochs_n
    # train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    # max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    # grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        # loss_list = []  # use this to record the loss value of each step
        # grad = []  # use this to record the loss gradient of each step
        # learning_curve[epoch] = 0  # maintain this to plot the training curve

        iters = 0
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            # record loss and gradients
            # loss_list.append(torch.round(loss.item(), decimals=6))
            # losses_list.append(loss.item())
            iters += 1
            if iters % 20 == 0:
                losses_list.append(loss.item())
            # grad.append(model.classifier[4].weight.grad.clone().to(torch.device('cpu')))
            ## --------------------

            optimizer.step()

        # losses_list.append(loss_list)
        # grads.append(grad)
        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))

        # learning_curve[epoch] /= batches_n
        # axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        # record accuracy on validation dataset
        val_accuracy_curve[epoch] = get_accuracy(val_loader, model, device)
        if max_val_accuracy < val_accuracy_curve[epoch]:
            max_val_accuracy = val_accuracy_curve[epoch]
            # max_val_accuracy_epoch = epoch
            # save new best model
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
        
        ## --------------------

    return losses_list

def main():
    # ## Constants (parameters) initialization
    # device_id = [0, 1, 2, 3]
    # num_workers = 4
    # batch_size = 128

    # add our package dir to path
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')

    # Make sure you are using the right device.
    # device_id = device_id
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(device)
    # print(torch.cuda.get_device_name(3))


    train_loader = get_cifar_loader(train_flag=True, shuffle=True)
    val_loader = get_cifar_loader(train_flag=False, shuffle=False)

    # Train your model
    # feel free to modify
    epo = 20
    loss_save_path = r'.\reports\results'
    grad_save_path = r'.\reports\results'

    set_random_seeds(seed_value=2020, device=device_name)
    model = VGG_A()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, init_lr=lr)
    loss = train(model, optimizer, criterion, train_loader, val_loader, scheduler=scheduler,epochs_n=epo, device=device)
    np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
    # np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    min_curve = []
    max_curve = []
    ## --------------------
    # Add your code
    # lr_list = [1e-4, 5e-4, 1e-3, 2e-3]
    lr_list = [1e-3, 5e-3, 1e-2, 5e-2]
    model = VGG_A()
    loss_list = []
    for lr in lr_list:
        # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, init_lr=lr)
        loss = train(model, optimizer, criterion, train_loader, val_loader, scheduler=scheduler, epochs_n=epo, device=device)
        loss_list.append(loss)

    for _ in range(len(loss_list[0])):
        losses = [it[_] for it in loss_list]
        min_curve.append(min(losses))
        max_curve.append(max(losses))

    min_curve_bn = []
    max_curve_bn = []
    model = VGG_A_BatchNorm()
    loss_list = []
    for lr in lr_list:
        # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        loss = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, device=device)
        loss_list.append(loss)

    for _ in range(len(loss_list[0])):
        losses = [it[_] for it in loss_list]
        min_curve_bn.append(min(losses))
        max_curve_bn.append(max(losses))
    ## --------------------

    # Use this function to plot the final loss landscape,
    # fill the area between the two curves can use plt.fill_between()
    def plot_loss_landscape(min_curve0, max_curve0, min_curve1, max_curve1):
        ## --------------------
        # Add your code
        plt.figure(figsize=(4, 4))
        plt.fill_between(range(len(min_curve0)), min_curve0, max_curve0, color='green', alpha=0.2, label='Standard VGG')
        plt.fill_between(range(len(min_curve1)), min_curve1, max_curve1, color='red', alpha=0.2, label='Standard VGG + BatchNorm')

        plt.xlabel('Steps')
        plt.ylabel('Loss Landscape')
        plt.title('Loss Landscape')
        plt.legend(loc='upper right')

        plt.show()
        ## --------------------
        pass

    plot_loss_landscape(min_curve, max_curve, min_curve_bn, max_curve_bn)


if __name__ == '__main__':
    main()