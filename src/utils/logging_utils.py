import sys
import json
from loguru import logger
from pathlib import Path

def setup_logging(log_dir: str = "logs/runs", level: str = "INFO"):
    log_path = Path(log_dir) / "run_{time}.log"
    json_log_path = Path(log_dir) / "run_{time}.jsonl"

    def write_json_log(msg):
        Path(json_log_path).open("a").write(json.dumps(msg.record, default=str) + "\n")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(sys.stderr, level=level)
    
    # Add file handler (human readable)
    logger.add(str(log_path), rotation="10 MB", level=level)
    
    # Add JSONL handler for structured logging
    logger.add(write_json_log, level=level)

def log_system_info():
    import platform
    import socket

    try:
        from src.utils.gpu_monitor import get_gpu_info
        gpu_info = get_gpu_info()
    except Exception as exc:
        gpu_info = {"cuda_available": False, "gpu_count": 0, "gpus": [], "error": str(exc)}
    
    info = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "gpu_info": gpu_info
    }
    logger.info(f"System Info: {json.dumps(info, indent=2)}")
    return info
