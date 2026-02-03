"""
Quick GPU detection test.
"""
import torch

print("="*60)
print("GPU DETECTION TEST")
print("="*60)

if torch.cuda.is_available():
    print(f"\n✅ GPU Available: YES")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print(f"\n❌ GPU Available: NO")
    print(f"   Running on CPU")

print("\n" + "="*60)
