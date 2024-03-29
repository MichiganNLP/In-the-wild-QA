# In-the-wild QA (WildQA) data and code

This repo contains the data and PyTorch code that accompanies our COLING 2022 paper:

[WildQA: In-the-Wild Video Question Answering](https://aclanthology.org/2022.coling-1.496)

[Santiago Castro](https://santi.uy/)+, [Naihao Deng](https://dnaihao.github.io/)+, Pingxuan Huang+,
[Mihai G. Burzo](https://sites.google.com/umich.edu/mburzo), and [Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/)
 
(+ equal contribution)

You can see more information at [the WildQA website](https://lit.eecs.umich.edu/wildqa/).

## Setup

With [Conda](https://docs.conda.io/en/latest/) installed, run:

```bash
conda env create
conda activate wildqa
```

Set `export WANDB_MODE=offline` if you don't want to use [Weights & Biases](https://wandb.ai/site) in your run.

## Data

Checkout the folder [`src/example_data/wildQA-data/`](src/example_data/wildQA-data).

## Run the code

For the methods presented in the paper, check out the Bash scripts under [`src/`](src).

## Citation

```bibtex
@inproceedings{castro-etal-2022-in-the-wild,
    title = "In-the-Wild Video Question Answering",
    author = "Castro, Santiago  and
      Deng, Naihao  and
      Huang, Pingxuan  and
      Burzo, Mihai G.  and
      Mihalcea, Rada",
    booktitle = "COLING",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.496",
    pages = "5613--5635",
}
```
