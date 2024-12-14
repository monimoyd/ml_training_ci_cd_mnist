import unittest
import sys
import numpy as np
import torch
import src.model
from src.model import Net
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from src.utils import count_parameters
#sys.path.append('..\\src')

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.train_dataset = MNIST(root='./data', train=True, 
                                 download=True, transform=self.transform)
        self.test_dataset = MNIST(root='./data', train=False, 
                                download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1000, 
                                     shuffle=False)
    
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 20000, "Model has too many parameters")

    def test_batch_normalization_used_in_model(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = Net().to(device)
        batch_normalization_used = "BatchNorm2d" in str(model)
        self.assertEqual(batch_normalization_used, True, "Batch Normalization is not used in model")

    def test_dropout_used_in_model(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = Net().to(device)
        dropout_used = "AdaptiveAvgPool2d" in str(model)
        self.assertEqual(dropout_used, True, "Dropout is not used in model")

    def test_gap_used_in_model(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = Net().to(device)
        gap_used = "AdaptiveAvgPool2d" in str(model)
        self.assertEqual(gap_used, True, "GAP is not used in model")



if __name__ == '__main__':
    unittest.main()