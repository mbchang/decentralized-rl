from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
from mnist.mnist_utils import load_mnist_datasets, shrink_mnist_dataset, float_mnist_dataset

class MNIST_CNN(nn.Module):
    """ Total Parameters: 193216 """
    def __init__(self, outdim):
        super(MNIST_CNN, self).__init__()
        self.outdim = outdim

        nc = 1
        ndf = 8
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.outdim, 4, 1, 0, bias=False))

    def forward(self, x):
        bsize = x.size(0)
        x = self.network(x)  # (bsize, num_outdim, 1, 1)
        x = x.view(bsize, self.outdim)
        return x


def embedded_train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    n = len(train_loader.data)
    num_batches = n // args.batch_size

    for batch_idx in range(num_batches):
        indices = np.random.randint(low=0, high=n, size=args.batch_size)
        data = train_loader.data[indices].to(device)
        target = train_loader.labels[indices].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), n,
                100. * batch_idx / n, loss.item()))


def embedded_test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    n = len(test_loader.data)
    num_batches = n // args.test_batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            indices = np.random.randint(low=0, high=n, size=args.test_batch_size)
            data = test_loader.data[indices].to(device)
            target = test_loader.labels[indices].to(device)

            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n,
        100. * correct / n))


def embedded_main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    mnist_orig = load_mnist_datasets('../data')
    mnist_shrunk = shrink_mnist_dataset(float_mnist_dataset(mnist_orig))

    model = MNIST_CNN(outdim=10).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        embedded_train(args, model, device, mnist_shrunk['train'], optimizer, epoch)
        embedded_test(args, model, device, mnist_shrunk['test'])
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "embedded_mnist_cnn_{}.pt".format(epoch))

if __name__ == '__main__':
    embedded_main()
