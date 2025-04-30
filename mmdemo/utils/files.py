import os
import random
from pathlib import Path


def create_tmp_dir(save_dir_prefix: str = None) -> Path:
    """
    Create a temporary directory and return
    a Path object to it. This is guaranteed
    to be a new and unique directory.
    """
    while True:
        dir = Path(f"output/{save_dir_prefix if save_dir_prefix else ''}/tmp_{int(random.random() * 10**6)}")
        try:
            os.makedirs(dir, exist_ok=False)
            return dir
        except FileExistsError:
            pass

def create_tmp_dir_with_featureName(featureName, save_dir_prefix: str = None) -> Path:
    """
    Create a temporary directory and return
    a Path object to it. This is guaranteed
    to be a new and unique directory.
    """
    while True:
        dir = Path(f"output/{save_dir_prefix if save_dir_prefix else ''}/tmp_{str(featureName)}_{int(random.random() * 10**6)}/")
        try:
            os.makedirs(dir, exist_ok=False)
            return dir
        except FileExistsError:
            pass
