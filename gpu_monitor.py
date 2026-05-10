import torch
from typing import Dict, Any

def get_gpu_info() -> Dict[str, Any]:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }
    
    if info["cuda_available"]:
        for i in range(info["gpu_count"]):
            gpu_props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": gpu_props.name,
                "total_memory_mb": gpu_props.total_memory // (1024**2),
                "major": gpu_props.major,
                "minor": gpu_props.minor
            })
    return info

def get_gpu_memory_usage() -> Dict[int, Dict[str, float]]:
    usage = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            usage[i] = {
                "allocated_mb": torch.cuda.memory_allocated(i) // (1024**2),
                "reserved_mb": torch.cuda.memory_reserved(i) // (1024**2)
            }
    return usage
