import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from autograd_engine import Value

def test_verification():
    print("--- Verify against PyTorch ---")
    
    a_val, b_val, c_val = 2.0, -3.0, 10.0
    
    # engine
    a = Value(a_val)
    b = Value(b_val)
    c = Value(c_val)
    out_custom = ((a * b + c)**2).relu().sum()
    out_custom.backward()
    
    a_t = torch.tensor(a_val, requires_grad=True)
    b_t = torch.tensor(b_val, requires_grad=True)
    c_t = torch.tensor(c_val, requires_grad=True)
    out_torch = ((a_t * b_t + c_t)**2).relu().sum()
    out_torch.backward()
    
    print(f"Custom Grads: a={a.grad}, b={b.grad}, c={c.grad}")
    print(f"Torch Grads:  a={a_t.grad.item()}, b={b_t.grad.item()}, c={c_t.grad.item()}")
    
    # difference
    max_diff = max(
        abs(a.grad - a_t.grad.item()),
        abs(b.grad - b_t.grad.item()),
        abs(c.grad - c_t.grad.item())
    )
    print(f"Max absolute diference: {max_diff}")
    assert max_diff < 1e-6, "Gradients do not match"
    print("PASSED: Max absolute diference < 1e-6\n")

if __name__ == "__main__":
    test_verification()