import os
import sys
import site
from pathlib import Path

def patch_pybasicbayes():
    # Find the pybasicbayes package directory
    pkg_name = "pybasicbayes"
    pkg_dir = None
    
    # Try current environment sites
    for s in site.getsitepackages() + [site.getusersitepackages()]:
        target = Path(s) / pkg_name
        if target.exists():
            pkg_dir = target
            break
            
    if not pkg_dir:
        print(f"Error: {pkg_name} not found in site-packages.")
        return False

    print(f"Found {pkg_name} at {pkg_dir}")

    # Patch 1: util/stats.py
    stats_py = pkg_dir / "util" / "stats.py"
    if stats_py.exists():
        content = stats_py.read_text()
        if "except ImportError:" in content and "RuntimeError" not in content:
            print(f"Patching {stats_py}")
            content = content.replace("except ImportError:", "except (ImportError, RuntimeError):")
            stats_py.write_text(content)
        else:
            print(f"{stats_py} already patched or pattern not found.")

    # Patch 2: distributions/gaussian.py
    gaussian_py = pkg_dir / "distributions" / "gaussian.py"
    if gaussian_py.exists():
        content = gaussian_py.read_text()
        old_pattern = "from numpy.core.umath_tests import inner1d"
        new_import = """try:
    from numpy.core.umath_tests import inner1d
except (ImportError, RuntimeError):
    import numpy as np
    def inner1d(a, b):
        return np.sum(a * b, axis=-1)"""
        
        if old_pattern in content and "try:" not in content:
            print(f"Patching {gaussian_py}")
            content = content.replace(old_pattern, new_import)
            gaussian_py.write_text(content)
        else:
            print(f"{gaussian_py} already patched or pattern not found.")

    return True

if __name__ == "__main__":
    if patch_pybasicbayes():
        print("Patches applied successfully.")
    else:
        sys.exit(1)
