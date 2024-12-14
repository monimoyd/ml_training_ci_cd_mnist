import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision import datasets, transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

