import torch
# import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.CNN import *


def main():
    # random seed
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using Device: {device}')

    # transforms for training and test data, augmentation for training
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # specifically for CIFAR-10
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=aug_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        persistent_workers=True, # persist worker process
        pin_memory=True, # speed up data transition from CPU to GPU
        prefetch_factor=2
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )

    # model
    # model = LeNetLike(input_channels=3, num_classes=10).to(device)
    # model = AdvLeNetLike(input_channels=3, num_classes=10).to(device)
    model = ResNetLike(input_channels=3, num_classes=10).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=5e-4)

    # learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    def compute_lr(epoch):
        if epoch <= 5:
            # warm up
            return 5e-2 * epoch / 5
        else:
            # cosine annealing -- take number of total epochs as value of cycle
            return 5e-2 * 0.5 * (1 + torch.cos(torch.tensor((epoch - 6) * torch.pi / 95)))

    # training
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        print(f'--------Epoch {epoch+1}--------')

        # obtain current learning rate
        curr_lr = compute_lr(epoch+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        # run training
        for i, (images, labels) in enumerate(trainloader, 0):
            # change device
            images, labels = images.to(device), labels.to(device)

            # set gradient to zero
            optimizer.zero_grad()

            # forward
            logits = model.forward(images)
            loss = criterion(logits, labels)

            # backward propagation
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # print information
            if (i+1) % 50 == 0:
                print(f'Iters: {i+1: 5d}  Loss: {loss.item(): .6f}')

        # test model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in testloader:
                images, labels = images.to(device), labels.to(device)

                logits = model.forward(images)
                _, prediction = torch.max(logits, 1)
                correct += torch.sum(torch.eq(prediction, labels)).item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f'Current Accuracy: {accuracy: .2f}%')

if __name__ == '__main__':
    main()
