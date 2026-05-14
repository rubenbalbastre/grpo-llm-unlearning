import torch

def main():
    # Check for CUDA availability
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA is available. GPU count: {n}")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"[{i}] {name} | VRAM: {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA is NOT available. Falling back to CPU.")
        # Create tensors on the selected device
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        # Perform an operation on the GPU
        z = x + y
        print("Hello, world! PyTorch with CUDA is working!")
        print("x:", x)
        print("y:", y)
        print("x + y:", z)

if __name__ == "__main__":
    main()