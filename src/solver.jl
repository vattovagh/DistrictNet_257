# Combinatorial layer: given edge/node weights, return a districting.

# TODO: capacitated_mst(graph, weights, capacity) -> forest/partition   (heuristic)
# TODO: local_search(city, partition, cost_fn) -> Partition             (improvement step)
# TODO: exact_solve(city, cost_fn) -> Partition                         (JuMP+GLPK, tiny instances only)
# TODO: partition_from_forest(forest) -> Partition

function compute_node_matrix(subgraphs, num_vertices)
    matrix = zeros(Int, length(subgraphs), num_vertices)
    for (i, subgraph) in enumerate(subgraphs)
        for node in subgraph
            matrix[i, node] = 1
        end
    end
    return matrix
end

function enumerate_connected_subgraphs(
    G::AbstractGraph,
    k::Int,
    m::Int,
    remove_depots::Bool = true,
)
    subgraphs = []
    non_root_nodes = collect(vertices(G))
    if remove_depots
        non_root_nodes = setdiff(vertices(G), nv(G))
    end
    for nb_nodes = k:m
        for selected_nodes in combinations(non_root_nodes, nb_nodes)
            SG, _ = induced_subgraph(G, selected_nodes)
            if is_connected(SG)
                push!(subgraphs, selected_nodes)
            end
        end
    end
    return subgraphs
end

function set_partitioning_optimizer(node_matrix, mst_costs, num_vertices, num_subgraphs)
    model = Model(GLPK.Optimizer)
    @variable(model, y[1:size(node_matrix, 1)], Bin)
    @objective(model, Min, sum(mst_costs[i] * y[i] for i = 1:size(node_matrix, 1)))

    for j = 1:num_vertices
        @constraint(
            model,
            sum(node_matrix[i, j] * y[i] for i = 1:size(node_matrix, 1)) == 1
        )
    end
    @constraint(model, sum(y) == num_subgraphs)
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return value.(y)
    else
        return 0
    end
end

function compute_subgraph_mst_cost(subgraphs, graph)
    mst_costs = []
    for subgraph_nodes in subgraphs
        subgraph, _ = induced_subgraph(graph, subgraph_nodes)
        mst_edges = kruskal_mst(subgraph)
        cost = sum([get_prop(subgraph, e, :weight) for e in mst_edges])
        root_weights = [get_prop(graph, nv(graph), node, :weight) for node in subgraph_nodes]
        cost += minimum(root_weights)
        push!(mst_costs, cost)
    end
    return Float64.(mst_costs)
end

function solve_districts(theta, instance, subgraphs)
    update_edge_weights!(instance.graph, reshape(-theta, :))
    graph = instance.graph
    n_districts = instance.num_blocks / instance.target_district_size

    node_matrix = compute_node_matrix(subgraphs, nv(graph) - 1)
    mst_costs = compute_subgraph_mst_cost(subgraphs, graph)
    y_opt = set_partitioning_optimizer(node_matrix, mst_costs, nv(graph) - 1, n_districts)
    y_opt == 0 && error("No feasible districting for $(instance.city_name)")
    return [Vector{Int64}(subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]
end

function cmst_exact_solver(theta; kwargs...)
    instance = kwargs[:instance]
    districts = solve_districts(theta, instance, kwargs[:subgraphs])
    solution = create_solution(instance, districts, Costloader([], []))
    return get_solution_edges(solution)
end

"""
function Exact_solve_instance(instance, costloader, solver_type, params)
    update_edge_weights!(
        instance.graph,
        solver_type == "CMST" ? params : rand(ne(instance.graph)),
    )
    graph = instance.graph
    lb, ub, k = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size
    subgraphs = enumerate_connected_subgraphs(graph, lb, ub, solver_type == "GNN" ? false : true)
    node_matrix = compute_node_matrix(subgraphs, nv(graph) - (solver_type == "GNN" ? 0 : 1))


    districts = []
    for i = 1:length(subgraphs)
        d = District(i, subgraphs[i], 0.0, true)
        push!(districts, d)
    end
    district_cost = [
        compute_district_cost(instance, d, costloader, solver_type, params) for
        d in districts
    ]

    y_opt = set_partitioning_optimizer(
        node_matrix,
        district_cost,
        nv(graph) - (solver_type == "GNN" ? 0 : 1),
        k,
    )
    districts = [Vector{Int64}(subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]
    solution = create_solution(instance, districts, costloader, solver_type, params)

    return solution
end
"""

function districting_exact_solver(instance, costloader)
    lower_bound, upper_bound, num_district = instance.min_district_size,
    instance.max_district_size,
    instance.num_blocks / instance.target_district_size
    # Compute unique subgraphs
    unique_subgraphs = enumerate_connected_subgraphs(instance.graph, lower_bound, upper_bound)
    node_matrix = compute_node_matrix(unique_subgraphs, nv(instance.graph) - 1)
    mapping = [
        [get_prop(instance.graph, j, :id) for j in unique_subgraphs[i]] for
        i = 1:length(unique_subgraphs)
    ]
    mapping_idx = [check_1d_in_2d(i, costloader.DistrictIds) for i in mapping]

    # Compute MST costs
    mst_costs = [
        mapping_idx[i] == nothing ? 1000 : costloader.Cost[mapping_idx[i]] for
        i = 1:length(mapping_idx)
    ]

    y_opt = set_partitioning_optimizer(node_matrix, mst_costs, nv(instance.graph) - 1, num_district)

    if y_opt == 0
        println("No solution found ",instance.city_name)
        return nothing, nothing
    else
        districts =[Vector{Int64}(unique_subgraphs[i]) for i = 1:length(y_opt) if y_opt[i] == 1]
        solution = create_solution(instance, districts, costloader, "Districting")
        return solution, unique_subgraphs
    end
end
