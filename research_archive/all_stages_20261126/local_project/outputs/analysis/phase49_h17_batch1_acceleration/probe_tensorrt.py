"""Check whether native TensorRT is present without building an engine."""
import ctypes
import importlib.util
import json
import shutil

result = {"python_module": bool(importlib.util.find_spec("tensorrt")),
          "trtexec": shutil.which("trtexec"), "native_dll": {}}
for name in ("nvinfer_10.dll", "nvinfer.dll"):
    try:
        ctypes.WinDLL(name)
        result["native_dll"][name] = "LOAD_OK"
    except OSError as exc:
        result["native_dll"][name] = f"UNAVAILABLE: {exc}"
print(json.dumps(result, indent=2))
