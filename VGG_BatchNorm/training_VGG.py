import torch
# import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from models.vgg import *

def main():
    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device != 'cpu':
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # model
    # model = VGG_A(inp_ch=3, num_classes=10, init_weights=True).to(device)
    model = VGG_A_BatchNorm(inp_ch=3, num_classes=10, init_weights=True).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=5e-4)

    # scheduler
    def compute_lr(epoch):
        if epoch <= 5:
            # warm up
            return 5e-2 * epoch / 5
        else:
            # cosine annealing -- take number of total epochs as value of cycle
            return 5e-2 * 0.5 * (1 + torch.cos(torch.tensor((epoch - 6) * torch.pi / 95)))

    # training
    loss_list = []
    acc_list = []
    best_acc = 0

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        print(f'--------Epoch {epoch+1}--------')

        # learning rate
        curr_lr = compute_lr(epoch + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # record loss
            loss_list.append(loss.item())

            if (i+1) % 100 == 0:
                print(f'Iters: {i+1: 6d}  Loss: {loss.item(): .6f}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in test_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model.forward(images)
                _, prediction = torch.max(logits, 1)
                correct += torch.sum(torch.eq(prediction, labels)).item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f'Current Accuracy: {accuracy: .2f}%')

        # record accuracy
        acc_list.append(correct / total)

        if accuracy > best_acc:
            print(f'Best Accuracy Updated. Accuracy: {best_acc: .2f} -> {accuracy: .2f}%')
            best_acc = accuracy

    # plot
    plt.figure(figsize=(8, 4))

    ax = plt.subplot(1, 2, 1)
    ax.plot(loss_list)
    ax.set_title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    ax = plt.subplot(1, 2, 2)
    ax.plot(acc_list)
    ax.set_title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()


if __name__ == '__main__':
    main()