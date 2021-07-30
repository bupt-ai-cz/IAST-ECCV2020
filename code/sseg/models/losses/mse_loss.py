import torch.nn as nn
import torch
import numpy as np
from ..registry import LOSS

@LOSS.register("MSELoss")
def MSE_loss(logits, labels):
    criterion = nn.MSELoss()
    return criterion(logits, labels)

