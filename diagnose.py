import sys
import os

print("=" * 60)
print("PYTORCH DLL ERROR DIAGNOSTIC")
print("=" * 60)

# Check 1: Python executable
print(f"\n1. Python: {sys.executable}")
print(f"   Version: {sys.version}")

# Check 2: Environment
print(f"\n2. Virtual Environment:")
print(f"   Active: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}")

# Check 3: PyTorch installation
print("\n3. PyTorch Status:")
try:
    import torch
    print(f"   ✅ PyTorch installed: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")

# Check 4: Diffusers
print("\n4. Diffusers Status:")
try:
    import diffusers
    print(f"   ✅ Diffusers installed: {diffusers.__version__}")
except Exception as e:
    print(f"   ❌ Diffusers error: {e}")

# Check 5: DLL dependencies
print("\n5. DLL Check:")
try:
    import torch
    torch_path = os.path.dirname(torch.__file__)
    print(f"   PyTorch location: {torch_path}")
    
    # Check for critical DLLs
    dll_files = ['torch_cpu.dll', 'torch_cuda_cpp.dll', 'c10.dll']
    for dll in dll_files:
        dll_path = os.path.join(torch_path, 'lib', dll)
        exists = os.path.exists(dll_path)
        print(f"   {dll}: {'✅ Found' if exists else '❌ Missing'}")
except Exception as e:
    print(f"   ❌ Error checking DLLs: {e}")

print("\n" + "=" * 60)
print("RECOMMENDED FIX:")
print("=" * 60)
print("\n1. Install Visual C++ Redistributable:")
print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
print("\n2. Reinstall PyTorch:")
print("   pip uninstall torch torchvision torchaudio -y")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("\n3. Restart computer and try again")
print("=" * 60)