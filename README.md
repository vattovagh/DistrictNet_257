# DistrictNet recreation

Own implementation (Julia) of "DistrictNet: Decision-Aware Learning for Geographical Districting" (arXiv 2412.08287).
`upstream/` is a local reference clone only (git-ignored); no code is copied from it.

Layout: `src/` library, `scripts/` entry points, `test/`, `docs/SPEC.md`, `data/`, `results/`.

Setup: Julia 1.10, `julia --project -e 'using Pkg; Pkg.instantiate()'` (Project.toml exists but might have things added to it).

Alternatively, you could run `julia` -> `]` -> `activate .` -> `instantiate`.   

# Data

All of this data is pulled from the original DistrictNet project.

`raw/` includes raw data from **real** cities.

`cities/` contains the training data that gets processed into the model.

`costs/` contains the predefined costs so re-generation is not needed.
