# In-the-wild QA

## Setup

With Conda and [Mamba](https://github.com/mamba-org/mamba) installed:

```bash
mamba env create
conda activate wildqa
```

Mamba is a Conda CLI drop-in replacement that's much faster.
It implements most operations. Some exceptions are `activate` and `deactivate`.

### Adding/changing packages

If you add or change packages, you can edit the `environment.yml` file and then run:

```bash
mamba env update
```
