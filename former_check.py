import torch
import sys

print("=" * 50)
print("SYSTEM COMPATIBILITY CHECK")
print("=" * 50)

# Python version
print(f"\n1. Python Version: {sys.version}")
print(f"   Required: 3.8 - 3.11")
python_ok = sys.version_info.major == 3 and 8 <= sys.version_info.minor <= 11
print(f"   Status: {'✅ Compatible' if python_ok else '❌ INCOMPATIBLE'}")

# PyTorch version
print(f"\n2. PyTorch Version: {torch.__version__}")
print(f"   Required: 2.0.0+")
pytorch_ok = torch.__version__.split('+')[0] >= "2.0.0"
print(f"   Status: {'✅ Compatible' if pytorch_ok else '❌ INCOMPATIBLE'}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n3. CUDA Available: {cuda_available}")
if cuda_available:
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Check compute capability (needs 3.7+)
    compute_cap = torch.cuda.get_device_capability(0)
    compute_ok = compute_cap[0] >= 3 and (compute_cap[0] > 3 or compute_cap[1] >= 7)
    print(f"   Status: {'✅ Compatible' if compute_ok else '❌ Too old (need 3.7+)'}")
else:
    print("   Status: ❌ No GPU detected (xFormers needs GPU)")

# Platform
import platform
print(f"\n4. Operating System: {platform.system()} {platform.release()}")
print(f"   Architecture: {platform.machine()}")

# Check if xFormers is already installed
print("\n5. xFormers Status:")
try:
    import xformers
    print(f"   ✅ Already installed: {xformers.__version__}")
except ImportError:
    print("   ❌ Not installed")

print("\n" + "=" * 50)
print("COMPATIBILITY SUMMARY")
print("=" * 50)

if cuda_available and python_ok and pytorch_ok:
    print("✅ Your system CAN install xFormers!")
    print("\nRecommended installation command:")
    cuda_version = torch.version.cuda
    if cuda_version.startswith("11.8"):
        print("pip install xformers --index-url https://download.pytorch.org/whl/cu118")
    elif cuda_version.startswith("12.1"):
        print("pip install xformers --index-url https://download.pytorch.org/whl/cu121")
    elif cuda_version.startswith("12.4"):
        print("pip install xformers --index-url https://download.pytorch.org/whl/cu124")
    else:
        print("pip install xformers")
else:
    print("❌ Your system may have compatibility issues")
    if not cuda_available:
        print("   → No CUDA GPU detected")
    if not python_ok:
        print("   → Python version incompatible")
    if not pytorch_ok:
        print("   → PyTorch version too old")

print("=" * 50)