# Keys are sorted node ids so node order doesn't matter.
function build_cost_lookup(costloader::Costloader)
    lookup = Dict{Vector{Int},Float64}()
    for (nodes, cost) in zip(costloader.DistrictIds, costloader.Cost)
        lookup[sort(nodes)] = cost
    end
    return lookup
end

function total_cost(districts::Vector{Vector{Int}}, lookup::Dict{Vector{Int},Float64})
    total = 0.0
    for nodes in districts
        key = sort(nodes)
        haskey(lookup, key) || error("District $key has no precomputed cost")
        total += lookup[key]
    end
    return total
end

function optimality_gap(cost::Real, reference_cost::Real)
    reference_cost > 0 || error("reference_cost must be positive, got $reference_cost")
    return 100 * (cost - reference_cost) / reference_cost
end

# `item` is one GraphFeaturesInstance; item.cost is the city's optimal cost.
function evaluate_city(model, item)
    theta = model(item.gnn_graph)
    districts = solve_districts(theta, item.instance, item.subgraphs)
    cost = total_cost(Vector{Vector{Int}}(districts), build_cost_lookup(item.costloader))
    return (cost = cost, optimal = item.cost, gap = optimality_gap(cost, item.cost))
end
