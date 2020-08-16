import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            if x.dim() != 4:
                assert x.dim() == 3
                x = x.unsqueeze(0)
                no_batch = True
            else:
                no_batch = False
            bsize = x.shape[0]
            theta = self.theta.expand(bsize, *self.theta.size())

            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid, padding_mode='border')
            if no_batch:
                x = x.squeeze()
            return x

class TranslateTransform(Transform):
    def __init__(self, dx, dy):
        Transform.__init__(self)
        self.theta = torch.Tensor(np.array(
            [[1, 0, dx],
             [0, 1, dy]]
             ))

class ScaleTransform(Transform):
    def __init__(self, scale):
        Transform.__init__(self)
        self.theta = torch.Tensor(np.array(
            [[scale, 0, 0],
             [0, scale, 0]]
             ))

class RotateTransform(Transform):
    def __init__(self, deg):
        Transform.__init__(self)
        radians = np.pi*deg/180
        self.theta = torch.Tensor(np.array(
            [[np.cos(radians), -np.sin(radians), 0],
             [np.sin(radians), np.cos(radians), 0]]
             ))

class TranslateLeftTransform(TranslateTransform):
    def __init__(self, translate_param):
        assert translate_param >= 0
        TranslateTransform.__init__(self, dx=translate_param, dy=0)

class TranslateRightTransform(TranslateTransform):
    def __init__(self, translate_param):
        assert translate_param >= 0
        TranslateTransform.__init__(self, dx=-translate_param, dy=0)

class TranslateUpTransform(TranslateTransform):
    def __init__(self, translate_param):
        assert translate_param >= 0
        TranslateTransform.__init__(self, dx=0, dy=translate_param)

class TranslateDownTransform(TranslateTransform):
    def __init__(self, translate_param):
        assert translate_param >= 0
        TranslateTransform.__init__(self, dx=0, dy=-translate_param)

class ScaleBigTransform(ScaleTransform):
    def __init__(self, scale_param):
        assert scale_param <= 1
        ScaleTransform.__init__(self, scale=scale_param)

class ScaleSmallTransform(ScaleTransform):
    def __init__(self, scale_param):
        assert scale_param <= 1
        ScaleTransform.__init__(self, scale=1.0/scale_param)

class RotateClockwiseTransform(RotateTransform):
    def __init__(self, rotate_param):
        assert rotate_param >= 0
        RotateTransform.__init__(self, deg=-rotate_param)

class RotateCounterClockwiseTransform(RotateTransform):
    def __init__(self, rotate_param):
        assert rotate_param >= 0
        RotateTransform.__init__(self, deg=rotate_param)
