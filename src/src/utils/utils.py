from typing import Any, MutableMapping

import torch
import numpy as np
import h5py


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_hdf5(path: str) -> MutableMapping[str, Any]:
    weights = {}
    keys = []
    with h5py.File(path) as file:
        file.visit(keys.append)
        for key in keys:
            weights[file[key].name.strip("/")] = file[key][:]
    return weights
