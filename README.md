# DistrictNet recreation

Own implementation (Julia) of "DistrictNet: Decision-Aware Learning for Geographical Districting" (arXiv 2412.08287).

Layout: `src/` and `dependencies` libraries, `scripts/` Our ML code, `data/`, `results/`.

Setup: Julia 1.10, `julia --project -e 'using Pkg; Pkg.instantiate()'` (Project.toml exists but might have things added to it).

Alternatively, you could run `julia` -> `]` -> `activate .` -> `instantiate`.   

# Data

All of this data is pulled from the original DistrictNet project.

`raw/` includes raw data from **real** cities.

`cities/` contains the training data that gets processed into the model.

`costs/` contains the predefined costs so re-generation is not needed.

# Code - Top Level Scripts

`train.jl` should encapsulate the whole training step of DistrictNet.

`evaluate.jl` holds out the model and tests on new data to confirm its quality.

`construct_solutions.jl` uses DistrictNet's Kruskal, greedy-merging, and district repair algorithms to convert the graph solution into real districts.

# Code - src/ and dependencies/ folders

These folders consists of code that is used in order to assist with tasks that are not machine-learning-related and are instead related to the specs of the original project. This includes work with graphs, solutions, district and solver data structures, and conversion into a feasible solution. `src/` consists of Julia code, and `dependencies/` consists of C++ code. 
