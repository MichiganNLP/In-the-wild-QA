import h5py

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            weights[f[key].name.strip("/")] = f[key].value
    return weights

def isfloat(ele):
    try:
        float(ele)
    except ValueError:
        return False
    return True
