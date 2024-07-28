"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np

import argparse
import models_2
import torch

def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)


def convert_fnames_to_v2(path: Path):
    """
    Converts filenames to conform to `v2` standard for knee data.

    For a file with name file1000.h5 in `path`, this script simply renames it
    to file1000_v2.h5. This is for submission to the public knee leaderboards.

    Args:
        path: Path with files to be renamed.
    """
    if not path.exists():
        raise ValueError("Path does not exist")

    for fname in path.glob("*.h5"):
        if not fname.name[-6:] == "_v2.h5":
            fname.rename(path / (fname.stem + "_v2.h5"))

            
def load_model(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
    args = argparse.Namespace(**{ **vars(state_dict["args"]), "no_log": True})

    model = models.build_model(args).to(device)
    model.load_state_dict(state_dict["model"][0])
    model.eval()
    return model


import torch




