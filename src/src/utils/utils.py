from typing import Any, MutableMapping

import h5py


def read_hdf5(path: str) -> MutableMapping[str, Any]:
    weights = {}
    keys = []
    with h5py.File(path) as file:
        file.visit(keys.append)
        for key in keys:
            weights[file[key].name.strip("/")] = file[key][:]
    return weights
