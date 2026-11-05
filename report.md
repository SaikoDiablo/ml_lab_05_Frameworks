# Lab 5 Report — Frameworks and Automatic Differentiation

## Goal
Build a scalar autograd engine from scratch, verify it against PyTorch's autograd, train a tiny model on `make_moons` with both engines to compare speed, and then compare eager vs. compiled/traced execution in PyTorch — including how a non-contiguous tensor affects operation speed.

## Method
Wrote a `Value` class in `autograd_engine.py` storing data, gradient, parent nodes, and a local backward rule. Implemented `__add__`, `__mul__`, `__pow__`, and `relu`, each building a graph node during the forward pass. Backward pass uses topological sort, sets the output gradient to 1.0, visits nodes in reverse, and **accumulates** gradients so branching works correctly.

Verification: compared my engine vs. `torch.autograd` on `((a*b + c)**2).relu().sum()`. Requirement was max absolute difference < 1e-6.

Training comparison: 2 → 8 → 1 MLP with sigmoid + BCE, trained with my engine and a PyTorch equivalent, median of 5 runs.

Execution modes: small 100 → 100 → 10 model. `torch.compile` failed with `InductorError` (no MSVC C++ build tools on Windows), so I used the lab-approved fallback `torch.jit.trace`.

Contiguity test: summed a non-contiguous tensor (`.t()`) vs. its `.contiguous()` copy, and checked `is_contiguous()`.

Seeds set to 42, timings median-based after warm-up.

## Results

**Autograd verification**

| Quantity | Custom | PyTorch | Max abs diff |
|---|---:|---:|---:|
| grad(a) | -24.0 | -24.0 | — |
| grad(b) | 16.0 | 16.0 | — |
| grad(c) | 8.0 | 8.0 | **0.0** |

**make_moons training (median of 5 runs)**

| Engine | Median time | Relative |
|---|---:|---:|
| Custom `Value` engine | 3.1480 s | 37.4× slower |
| PyTorch | 0.0842 s | 1× |

**Eager vs traced execution**

| Mode | Time (50 forward passes) | Speedup |
|---|---:|---:|
| Eager | 0.0091 s | 1× |
| `torch.jit.trace` | 0.0074 s | **1.24×** |
| `torch.compile` | Failed — `InductorError` (MSVC C++ build tools missing) | — |

**Contiguity**

| Tensor | `is_contiguous()` | Sum time |
|---|---|---:|
| `a` (normal) | `True` | — |
| `a.t()` (transposed) | `False` | 0.2637 s |
| `a.t().contiguous()` | `True` | 0.2718 s |

The contiguous copy was slightly *slower* here — at this small size, the cost of copying dominates the memory-access savings.

## Conclusions
1. **My autograd engine matched PyTorch exactly (max abs diff 0.0).** That confirms the topological backward pass and gradient accumulation are correct, including the branching in `a*b + c`.
2. **PyTorch was ~37× faster on the same tiny model.** My engine loops over Python scalar objects with a Python call per operation; PyTorch runs vectorized C++ ops with BLAS matmul. Basically framework overhead is dominated by the backend, not the math.
3. **Tracing gave a 1.24× speedup over eager, and the contiguity test was a reminder that theory doesn't always show up in micro-benchmarks.** Tracing helps most when a model has lots of small sequential ops (kernel fusion reduces Python/launch overhead); for tiny models the compile cost can outweigh the gain. Likewise, contiguous memory is supposed to be faster, but at small sizes the copy needed to make it contiguous cost more than the memory-access savings — the theory only pays off at larger sizes.
