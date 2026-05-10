import subprocess
from typing import List, Tuple, Optional

def run_command(cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout.decode() if e.stdout else "", "Timeout expired"
    except Exception as e:
        return -2, "", str(e)
