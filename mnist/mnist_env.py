from collections import namedtuple
import itertools
import torch
from torch.utils.data import Dataset

from mnist.embedded_mnist import MNIST_CNN
from mnist.mnist_utils import get_mnist_datasets
from starter_code.modules.affine_transformations import TranslateLeftTransform, TranslateRightTransform, TranslateUpTransform, TranslateDownTransform, ScaleBigTransform, ScaleSmallTransform, RotateClockwiseTransform, RotateCounterClockwiseTransform

mnist_classifier = MNIST_CNN(outdim=10)
mnist_classifier.load_state_dict(torch.load('mnist/mnist_classifier.pt'))
mnist_classifier.eval()  # need to do eval mode because we had done dropout before

def zero_one_loss(x, y):
    return torch.tensor([float(torch.allclose(x,y))])

def mnist_loss_01(image, target_label):
    predicted_logits = mnist_classifier(image)
    pred = predicted_logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target_label.view_as(pred)).sum()
    return correct

def get_affine_transforms():

    rotate_param = 60
    scale_param = 0.6
    translate_small_param = 0.38
    translate_normal_param = 0.29
    translate_big_param = 0.2

    def rotate_left(img):
        return RotateCounterClockwiseTransform(rotate_param=rotate_param)(img)

    def rotate_right(img):
        return RotateClockwiseTransform(rotate_param=rotate_param)(img)

    def scale_small(img):
        return ScaleSmallTransform(scale_param=scale_param)(img)

    def scale_big(img):
        return ScaleBigTransform(scale_param=scale_param)(img)

    def translate_up_small(img):
        return TranslateUpTransform(translate_param=translate_small_param)(img)

    def translate_down_small(img):
        return TranslateDownTransform(translate_param=translate_small_param)(img)

    def translate_left_small(img):
        return TranslateLeftTransform(translate_param=translate_small_param)(img)

    def translate_right_small(img):
        return TranslateRightTransform(translate_param=translate_small_param)(img)


    def translate_up_normal(img):
        return TranslateUpTransform(translate_param=translate_normal_param)(img)

    def translate_down_normal(img):
        return TranslateDownTransform(translate_param=translate_normal_param)(img)

    def translate_left_normal(img):
        return TranslateLeftTransform(translate_param=translate_normal_param)(img)

    def translate_right_normal(img):
        return TranslateRightTransform(translate_param=translate_normal_param)(img)


    def translate_up_big(img):
        return TranslateUpTransform(translate_param=translate_big_param)(img)

    def translate_down_big(img):
        return TranslateDownTransform(translate_param=translate_big_param)(img)

    def translate_left_big(img):
        return TranslateLeftTransform(translate_param=translate_big_param)(img)

    def translate_right_big(img):
        return TranslateRightTransform(translate_param=translate_big_param)(img)

    PrimitiveAffineTransform = namedtuple('PrimitiveAffineTransform', 
        (
                'rotate_left',
                'rotate_right',

                'translate_up_normal',
                'translate_down_normal',
                'translate_left_normal',
                'translate_right_normal',
            ))

    affine_transforms = PrimitiveAffineTransform(
            rotate_left=rotate_left,
            rotate_right=rotate_right,

            translate_up_normal=translate_up_normal,
            translate_down_normal=translate_down_normal,
            translate_left_normal=translate_left_normal,
            translate_right_normal=translate_right_normal,

        )
    return affine_transforms


class MNISTDataset:
    datasets = get_mnist_datasets('data')
    affine_transforms = get_affine_transforms()

class MentalRotation(Dataset):
    def __init__(self):
        self.transformation_combinations = list(itertools.product(
            [MNISTDataset.affine_transforms.rotate_left, MNISTDataset.affine_transforms.rotate_right], 
            [MNISTDataset.affine_transforms.translate_up_normal, MNISTDataset.affine_transforms.translate_down_normal, MNISTDataset.affine_transforms.translate_left_normal, MNISTDataset.affine_transforms.translate_right_normal]))  # (1 x 2 x 4)

        self.inputs, self.outputs = self.apply_transformation_combinations(
            data=MNISTDataset.datasets['train'].data, 
            labels=MNISTDataset.datasets['train'].labels)
        self.in_dim = (1, 64, 64)
        self.out_dim = 0  # dummy value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        inp: (1, C, H, W)
        out: (1)
        """
        return self.inputs[idx:idx+1], self.outputs[idx:idx+1]

    def apply_transformations(self, x, transformations):
        for transformation in transformations:
            x = transformation(x)
        return x

    def apply_transformation_combinations(self, data, labels):
        inputs = []
        outputs = []
        for transformations in self.transformation_combinations:
            print('Applying transformation: {}'.format(transformations))
            transformed_inputs = self.apply_transformations(data, transformations)  # make sure that we are making a copy
            inputs.append(transformed_inputs)
            outputs.append(labels)
        inputs = torch.cat(inputs, dim=0)
        outputs = torch.cat(outputs, dim=0)
        return inputs, outputs
