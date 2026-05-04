import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(10, 10))
    def forward(self, x):
        return x @ self.w

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = DummyLayer()
        self.use_checkpointing = True
    def forward(self, x):
        if self.use_checkpointing:
            return checkpoint(self.layer, x, use_reentrant=False)
        return self.layer(x)

model = DummyModel()
x = torch.randn(10, 10, requires_grad=False)
y = model(x)
loss = y.sum()
try:
    loss.backward()
    print("Gradients with requires_grad=False input:")
    print(model.layer.w.grad)
except Exception as e:
    print(f"Backward failed: {e}")
