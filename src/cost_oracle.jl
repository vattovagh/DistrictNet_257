# Ground-truth cost of a single district (stochastic routing cost).

# TODO: district_cost(city, district; n_scenarios, rng) -> Float64
#   first version: simple TSP heuristic (nearest neighbour + 2-opt) averaged over
#   random customer scenarios
# TODO: precompute_costs(city, district_size) -> Dict{Vector{Int},Float64}
# TODO: compare against upstream/data/tspCosts as a sanity check (not a dependency)
