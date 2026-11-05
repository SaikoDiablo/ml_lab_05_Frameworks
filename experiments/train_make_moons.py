import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
from sklearn.datasets import make_moons
from autograd_engine import Value

class TinyMLP:
    def __init__(self):
        # 2-8-1
        self.w1 = [Value(np.random.randn()) for _ in range(16)] 
        self.b1 = [Value(0.0) for _ in range(8)]
        self.w2 = [Value(np.random.randn()) for _ in range(8)] 
        self.b2 = Value(0.0)

    def __call__(self, x):
        hidden = []
        for i in range(8):
            z = self.w1[i*2] * x[0] + self.w1[i*2+1] * x[1] + self.b1[i]
            hidden.append(z.relu())
        
        out = self.b2
        for i in range(8):
            out = out + self.w2[i] * hidden[i]
        
        s = 1.0 / (1.0 + math.exp(-out.data))
        return Value(s)

class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 8)
        self.layer2 = torch.nn.Linear(8, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(x))

def run_custom_engine(X, y, epochs=100, lr=0.1):
    model = TinyMLP()
    for epoch in range(epochs):
        y_pred = [model(x) for x in X]
        loss = Value(0.0)
        for yt, yp in zip(y, y_pred):
            loss = loss + (yp - yt)**2 
        loss.backward()
        
        model.b2.data -= lr * model.b2.grad
        for p in model.w1 + model.b1 + model.w2:
            p.data -= lr * p.grad
            p.grad = 0.0 
        model.b2.grad = 0.0

def run_torch(X_t, y_t, epochs=100, lr=0.1):
    model = TorchMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = criterion(y_pred, y_t)
        loss.backward()
        optimizer.step()

import math 

def task5_make_moons():
    print("--- Training tiny model on make_moons ---")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
 
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    custom_times = []
    torch_times = []

    for run in range(5):
        start = time.time()
        run_custom_engine(X, y)
        custom_times.append(time.time() - start)

        start = time.time()
        run_torch(X_t, y_t)
        torch_times.append(time.time() - start)

    median_custom = np.median(custom_times)
    median_torch = np.median(torch_times)
    
    print(f"Custom Engine Median Time : {median_custom:.4f} sec")
    print(f"PyTorch Median Time: {median_torch:.4f} sec")
    print(f"PyTorch is faster by: {median_custom/median_torch:.1f}x\n")

if __name__ == "__main__":
    task5_make_moons()