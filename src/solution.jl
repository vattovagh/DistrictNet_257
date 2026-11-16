function get_solution_edges(solution::Solution)
    instance, graph = solution.instance, solution.instance.graph
    g_edges = [(src(edge), dst(edge)) for edge in edges(graph)]
    edge_in_solution = zeros(Bool, length(edges(graph)))

    for district in solution.districts
        nodes = deepcopy(district.nodes)
        sub_graph, original = induced_subgraph(graph, nodes)

        for edge in prim_mst(sub_graph)
            idx = find_edge_index((original[src(edge)], original[dst(edge)]), g_edges)
            edge_in_solution[idx] = true
        end

        idx_min = argmin([
            get_prop(graph, instance.num_blocks + 1, node, :weight) for node in nodes
        ])
        idx = find_edge_index((nodes[idx_min], instance.num_blocks + 1), g_edges)
        edge_in_solution[idx] = true
    end

    return Int.(edge_in_solution)
end

find_edge_index(edge, edges) = findfirst(e -> e == edge || reverse(e) == edge, edges)

function create_solution(
    instance::Instance,
    districts::Vector{Vector{Int64}},
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)::Solution
    new_districts = [
        create_district(i, nodes, instance, costloader, mode, model) for
        (i, nodes) in enumerate(districts)
    ]
    cost = compute_solution_cost(new_districts)
    is_feasible = all(district.is_feasible for district in new_districts)
    if mode == "GNN"
        blocks_district_ids =
            [get_district_id(new_districts, node) for node = 1:nv(instance.graph)]
    else
        blocks_district_ids =
            [get_district_id(new_districts, node) for node = 1:nv(instance.graph)-1]
    end
    return Solution(
        instance,
        new_districts,
        cost,
        length(districts),
        is_feasible,
        blocks_district_ids,
    )
end

function compute_solution_cost(districts::Array{District})::Float64
    return sum(district.cost for district in districts)
end

function compute_solution_cost(solution::Solution)::Float64
    return sum(district.cost for district in solution.districts)
end

function is_solution_feasible(solution::Solution)::Bool
    return all(district.is_feasible for district in solution.districts)
end

function get_district_id(districts::Array{District}, node::Int)::Int
    for district in districts
        if node in district.nodes
            return district.id
        end
    end
    error("The node $node is not in any district")
end

function get_border_nodes(solution::Solution, id1::Int, id2::Int)
    district1, district2 = solution.districts[id1], solution.districts[id2]
    border1, border2 = Set{Int}(), Set{Int}()

    for node1 in district1.nodes
        for node2 in district2.nodes
            if has_edge(solution.instance.graph, node1, node2)
                push!(border1, node1)
                push!(border2, node2)
            end
        end
    end

    return isempty(border1) ? () : (border1, border2)
end
