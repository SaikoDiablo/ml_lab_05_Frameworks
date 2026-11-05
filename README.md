# Lab 5: Frameworks and Automatic Differentiation

## How to run
1. Install the dependencies: `pip install -r requirements.txt`
2. Run everything at once: `python main.py`

## Graded Evidence

### 1. Autograd Engine Verification
For this part I built a small scalar autograd engine in `autograd_engine.py` that stores the data, the gradient, and the local backward function for each node. To check it actually works, I compared my gradients against PyTorch's `autograd` using the formula `((a*b + c)**2).relu().sum()`.

The maximum absolute difference between my engine and PyTorch was **0.0**, so the math checks out

### 2. Training on make_moons
I trained a tiny 2 - 8 - 1 MLP on `make_moons` with both my custom engine and PyTorch, and timed each over 5 runs.-

- **Custom engine (median):** 3.1480 sec
- **PyTorch (median):** 0.0842 sec

So PyTorch was about **37.4x faster**. That makes sense to me now my engine is just pure Python looping over scalar objects, while PyTorch uses a C++ backend and vectorized matrix operations under the hood. Basically every operation in my engine has Python overhead that PyTorch avoids.

### 3. Eager vs Compiled Execution
My Windows setup doesn't have the MSVC C++ build tools, so `torch.compile` failed with an `InductorError` (it needs a C++ compiler to build the kernels). The lab guide says to fall back to tracing/scripting in that case, so I used `torch.jit.trace` instead.

- **Eager time:** 0.0091 sec
- **Traced time:** 0.0074 sec
- **Speedup:** 1.24x

(One thing I noticed: PyTorch threw a `FutureWarning` saying `torch.jit.trace` is deprecated in favor of `torch.compile`. Since compilation isn't available on my machine anyway, I just left it as is for this experiment)

**Tensor contiguity:**
For this test I created a normal tensor and its transpose. The transposed one is non-contiguous (`is_contiguous()` returned `False`), and summing it took **0.2637s**, while the contiguous version took about **0.2718s**. Honestly, the contiguous one was *slightly slower* here, which I think is because on this small size the copy needed to make it contiguous costs more than the speedup it gives. In theory non-contiguous tensors are slower because memory access jumps around and causes cache misses, but for a micro-benchmark like this the overhead of copying can hide that.

**Short note on when compilation helps:**
From what I understand, compilation helps most when a model has lots of small operations back-to-back, because the compiler fuses them into one kernel and removes Python overhead. It's mainly useful during training loops of larger models. For very small models or a single forward pass, the compile time itself usually outweighs any speedup.