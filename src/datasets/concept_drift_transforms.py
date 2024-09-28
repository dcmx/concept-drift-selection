import torch
from torchvision.transforms import GaussianBlur

class AddGaussianNoise(object):
    rand_vec = None
    def __init__(self, mean=0., std=1., mode='hard'):
        self.std = std
        self.mean = mean
        self.mode = mode
        #self.rand_vec = torch.randn(24)

    def __call__(self, tensor):
        if AddGaussianNoise.rand_vec is None:
            AddGaussianNoise.rand_vec = torch.rand(tensor.size())
        if self.mode == 'soft':
            tensor = GaussianBlur(7, sigma=(2 * self.std))(tensor)
            # tensor = torch.clamp(tensor + ((self.rand_vec -0.5) * (0.5 * self.std) + self.mean), 0, 1)
            #tensor = ElasticTransform()
            # CIFAR
            return tensor
        if self.mode == 'hard':
            tensor = GaussianBlur(11, sigma=(5 * self.std))(tensor)
            #tensor = torch.clamp(tensor + ((torch.rand(tensor.size())- 0.5) * (0.5 * self.std) + self.mean), 0, 1)
            # MNIST
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)