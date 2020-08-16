from collections import namedtuple
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

MNISTDataset = namedtuple('MNISTDataset', ('data', 'labels'))

def load_mnist_datasets(root='data', extrap=False):
    train_dataset = datasets.MNIST(root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valtest_dataset = datasets.MNIST(root='data', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
               ]))
    # now you should divide into groups
    numtest = int(len(valtest_dataset) / 2)
    mnist_datasets = {
        'train': MNISTDataset(
            data=train_dataset.data.unsqueeze(1), 
            labels=train_dataset.targets),
        'val': MNISTDataset(
            data=valtest_dataset.data[:numtest].unsqueeze(1), 
            labels=valtest_dataset.targets[:numtest]),
        'test': MNISTDataset(
            data=valtest_dataset.data[numtest:].unsqueeze(1), 
            labels=valtest_dataset.targets[numtest:]),
    }
    if extrap:
        mnist_datasets['extrapval'] = (valtest_dataset.data[numtest:], valtest_dataset.targets[numtest:])
    return mnist_datasets

def place_subimage_in_background(subimage, bkgd_dim):
    assert subimage.dim() == 4
    bsize, c, h, w = subimage.shape
    bkgd_h, bkgd_w = bkgd_dim

    bkgd = torch.zeros((bsize, c, bkgd_h, bkgd_w))

    from_top_limit = bkgd_h-h+1
    from_left_limit = bkgd_w-w+1

    top = from_top_limit//2
    left = from_left_limit//2

    bkgd[:, :, top:top+h, left:left+w] = subimage
    return bkgd

def shrink_mnist_dataset(mnist_orig):
    mnist_shrunk = {}
    for k in mnist_orig.keys():
        v_data, v_labels = mnist_orig[k]
        mnist_shrunk[k] = MNISTDataset(data=place_subimage_in_background(v_data, (64, 64)), 
            labels=v_labels)
    return mnist_shrunk

def float_mnist_dataset(mnist_orig):
    mnist_float = {}
    for k in mnist_orig.keys():
        v_data, v_labels = mnist_orig[k]
        mnist_float[k] = MNISTDataset(data=v_data.float()/255.0, labels=v_labels)
    return mnist_float

def get_mnist_datasets(root):
    return shrink_mnist_dataset(float_mnist_dataset(load_mnist_datasets(root)))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    def test_load_mnist_datasets():
        mnist_datasets = load_mnist_datasets()
        for key, value in mnist_datasets.items():
            print(key, value.data.shape, value.labels.shape)

    def test_shrink_mnist_dataset():
        mnist_datasets = load_mnist_datasets()
        mnist_float = float_mnist_dataset(mnist_datasets)
        mnist_shrunk = shrink_mnist_dataset(mnist_float)
        nplots = 5
        fig = plt.figure()
        for i in range(1, nplots+1):
            ax = fig.add_subplot(nplots, 2, 2*i-1)
            ax.imshow(mnist_datasets['train'].data[i][0])
            ax = fig.add_subplot(nplots, 2, 2*i)
            ax.imshow(mnist_shrunk['train'].data[i][0])
        plt.savefig('embed.png')
        plt.close()

    test_load_mnist_datasets()
    test_shrink_mnist_dataset()


