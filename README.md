# MonsterLM

A versatile, fast and unbiased method for estimation of gene-by-environment interaction effects on biobank-scale datasets.

[Paper](https://www.nature.com/articles/s41467-023-40913-7)

## Installation

Requires the [Rust programming language](https://rust-lang.org).

```r
devtools::install_github("mrvillage/MonsterLM")
```

## Usage

The main function is `monsterlm::monsterlm`.
- The first argument is the directory where the data is stored.
- The second argument is the type, either `"continuous"` or `"dichtomous"`.
- MonsterLM expects the data be stored in files `outcomes.rkyv.gz`, `exposures.rkyv.gz`, `age.rkyv.gz`, `sex.rkyv.gz`, `pcs.rkyv.gz`, and `chr_{}_set_{}.rkyv.gz` for 22 chromosomes, with sets ranging from 1 to the respective entry in the array `[4, 4, 4, 3, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]`.
- The function returns a data frame of results with the fields `model`, `outcome`, `exposure`, `pred_n`, `n`, `est_adj`, `lci`, `uci`, `variance_total`, `standard_deviation`.

```r
monsterlm::monsterlm("data_dir", "continuous")
```
