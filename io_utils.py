import os
import shutil
from pathlib import Path
from typing import List, Union

def ensure_dir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)

def list_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [f for f in directory.iterdir() if f.is_file()]
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return files

def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    shutil.copy2(src, dst)

def read_text(path: Union[str, Path]) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: Union[str, Path], content: str):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
