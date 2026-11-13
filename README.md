# DistrictNet recreation

Own implementation (Julia) of "DistrictNet: Decision-Aware Learning for Geographical Districting" (arXiv 2412.08287).
`upstream/` is a local reference clone only (git-ignored); no code is copied from it.

Layout: `src/` library, `scripts/` entry points, `test/`, `docs/SPEC.md`, `data/`, `results/`.
Setup: Julia 1.10, `julia --project -e 'using Pkg; Pkg.instantiate()'` (Project.toml exists but might have things added to it).
Alternatively, you could run `julia` -> `]` -> `activate .` -> `instantiate`.   
