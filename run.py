#!/usr/bin/env python3
import os, sys

# On Windows, add PyTorch's bundled CUDA 11.8 DLLs to the DLL search path.
# This allows onnxruntime-gpu to find cudart64_110.dll, cudnn64_8.dll etc.
# even when the system has a different CUDA version installed (e.g. CUDA 13.x).
if sys.platform == 'win32':
    # sys.executable = venv/Scripts/python.exe  → venv root is two levels up
    _scripts = os.path.dirname(sys.executable)           # venv/Scripts
    _venv    = os.path.dirname(_scripts)                  # venv
    _torch_lib = os.path.join(_venv, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.isdir(_torch_lib):
        os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(_torch_lib)
        except Exception:
            pass

from modules import core

if __name__ == '__main__':
    core.run()
