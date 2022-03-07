# In-the-wild QA

## Setup

With Conda and [Mamba](https://github.com/mamba-org/mamba) installed:

```bash
mamba env create
mamba activate wildqa
```

Mamba is a Conda CLI drop-in replacement that's much faster.

### Adding/changing packages

If you add or change packages, you can edit the `environment.yml` file and then run:

```bash
mamba env update
```

### Issues with mamba

Sometimes `mamba` changes `torch` version to CPU. If that happens, do

```bash
mamba install pytorch::pytorch
```
