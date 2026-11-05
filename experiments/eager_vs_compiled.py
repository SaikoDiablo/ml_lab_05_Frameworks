import time
import torch

def task6_compilation():
    print("--- Eager vs Compiled Execution ---")
    
    # def small model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )
    x = torch.randn(64, 100) 
    
    for _ in range(5): model(x)
    
    # eager ex
    start = time.time()
    for _ in range(50): model(x)
    eager_time = time.time() - start
    print(f"Eager time: {eager_time:.4f}s")
    
    # Compiled
    print("Attempting torch.compile...")
    try:
        compiled_model = torch.compile(model)
        for _ in range(5): compiled_model(x) # Warmup/Compile
        start = time.time()
        for _ in range(50): compiled_model(x)
        compiled_time = time.time() - start
        print(f"Compiled time: {compiled_time:.4f}s")
        print(f"Speedup: {eager_time/compiled_time:.2f}x")
    except Exception as e:
        # fallback to JIT 
        print(f"Compilation failed ({type(e).__name__}). Falling back to torch.jit.trace...")
        
        traced_model = torch.jit.trace(model, x)
        for _ in range(5): traced_model(x)
        
        start = time.time()
        for _ in range(50): traced_model(x)
        traced_time = time.time() - start
        print(f"Traced time: {traced_time:.4f}s")
        print(f"Speedup: {eager_time/traced_time:.2f}x")

    # check contiguity
    print("\n--- Contiguity Check ---")
    a = torch.randn(1000, 1000)
    b = a.t() 
    print(f"Is 'a' contiguous? {a.is_contiguous()}")
    print(f"Is 'b' (transposed) contiguous? {b.is_contiguous()}")

    start = time.time()
    for _ in range(1000): _ = b.sum()
    non_contig_time = time.time() - start

    b_contig = b.contiguous() 
    start = time.time()
    for _ in range(1000): _ = b_contig.sum()
    contig_time = time.time() - start
    
    print(f"Non-contigous sum time: {non_contig_time:.4f}s")
    print(f"Contiguous sum time: {contig_time:.4f}s\n")