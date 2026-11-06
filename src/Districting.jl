module Districting

using Graphs, Random, Statistics

include("instance.jl")
include("data.jl")
include("cost_oracle.jl")
include("solver.jl")
include("gnn.jl")
include("training.jl")
include("baselines.jl")
include("evaluation.jl")

end # module
