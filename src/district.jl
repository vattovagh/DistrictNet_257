function create_district(
    i::Int,
    nodes::Vector{Int64},
    instance::Instance,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)::District
    cost = compute_district_cost(
        instance,
        District(i, nodes, 0.0, false),
        costloader,
        mode,
        model,
    )
    feasible = is_district_feasible(instance, District(i, nodes, cost, false))
    return District(i, nodes, cost, feasible)
end

function is_district_feasible(instance::Instance, district::District)
    n_nodes = length(district.nodes)
    return n_nodes >= instance.min_district_size &&
           n_nodes <= instance.max_district_size &&
           is_subgraph_connected(instance.graph, district.nodes)
end

function compute_district_cost(
    instance::Instance,
    district::District,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    if mode == "Districting"
        return compute_cost_with_precomputed_data(instance, district, costloader)
    elseif mode == "CMST"
        return compute_cost_via_CMST(instance, district)
    else
        error("Invalid mode")
    end
end

function compute_cost_with_precomputed_data(
    instance::Instance,
    district::District,
    costloader::Costloader,
)
    g = instance.graph
    nodes = district.nodes
    original_nodes = [get_prop(g, j, :id) for j in nodes]
    idx = check_1d_in_2d(original_nodes, costloader.DistrictIds)
    return idx === nothing ? compute_cost_via_SAA(instance, original_nodes) :
           costloader.Cost[idx]
end

function compute_cost_via_SAA(instance::Instance, nodes::Array{Int})
    g = instance.graph
    nodes = nodes .- 1
    formatted_nodes = "{" * join(nodes, ", ") * "}"
    target = string( instance.city_name, "_", instance.depot, "_", instance.num_blocks, "_", instance.target_district_size, "_", NB_SCENARIO)
    return EVmain(formatted_nodes, target)
end

function compute_cost_via_CMST(instance::Instance, district::District)
    nodes = district.nodes
    if length(nodes) <= 1
        return PENALTY * instance.max_district_size
    end
    sub_graph, _ = induced_subgraph(instance.graph, nodes)
    cost = sum(get_prop(sub_graph, edge, :weight) for edge in prim_mst(sub_graph))
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALTY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost + compute_depot_cost(instance, nodes)
end

function compute_depot_cost(instance::Instance, nodes::Array{Int})
    return minimum([
        get_prop(instance.graph, instance.num_blocks + 1, node, :weight) for node in nodes
    ])
end
