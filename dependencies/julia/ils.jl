function run_kruskal(instance::Instance)
    """
    Kruskal computes a minimum spanning tree using Kruskal's algorithm.

    Parameters:
    - `instance::Instance`: The problem instance

    Returns:
    - `uf::UnionFind`: A UnionFind object representing the minimum spanning tree
    """
    g = instance.graph
    G = deepcopy(g)
    nb = instance.num_blocks
    g_edges =
        [Edge(src(i), dst(i), get_prop(G, i, :weight)) for i in edges(G) if src(i) != nb +1 &&
                                                                                         dst(
            i,
        ) != nb +1]
    sorted_edges = sort(g_edges, by = x -> x.weight)
    uf = UnionFind.UnionFinder(nb)
    for e in sorted_edges
        u, v, w = e.src, e.dst, e.weight
        if (find!(uf, u) != find!(uf, v)) &&
           (size!(uf, u) + size!(uf, v) <= instance.target_district_size)
            UnionFind.union!(uf, u, v)
        end
    end

    return uf
end


"""
findNeighborDistricts finds the neighboring districts of each district.

Parameters:
- `instance::Instance`: The problem instance
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts

Returns:
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping a pair of neighboring districts to the number of nodes in the pair
"""
function findNeighborDistricts(instance::Instance, districts::Vector{Vector{Int}})
    neighborPairs = Dict{Tuple{Int,Int},Int}()
    g = instance.graph
    for i = 1:length(districts)
        for u in districts[i]
            for v in neighbors(g, u)
                if (v == instance.num_blocks + 1)
                    continue
                end
                found_v = false
                for j = (i+1):length(districts)
                    if v in districts[j]
                        node_sum = length(districts[i]) + length(districts[j])
                        neighborPairs[(i, j)] = get(neighborPairs, (i, j), node_sum)
                        found_v = true
                        break
                    end
                end
                if found_v
                    break
                end
            end
        end
    end

    return neighborPairs
end


"""
mergeDistricts merges two districts.

Parameters:
- `uf::UnionFinder`: A UnionFind object representing the minimum spanning tree
- `d1::Vector{Int}`: A vector of nodes representing the first district
- `d2::Vector{Int}`: A vector of nodes representing the second district
"""
function mergeDistricts(uf::UnionFinder, d1::Vector{Int}, d2::Vector{Int})
    for u in d1
        for v in d2
            UnionFind.union!(uf, u, v)
        end
    end
end


"""
groupBlocks groups the nodes in the minimum spanning tree into districts.

Parameters:
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function groupBlocks(uf::UnionFinder)
    cf = CompressedFinder(uf)
    groups = cf.groups
    district = Vector{Vector{Int}}(undef, groups)
    for i = 1:groups
        district[i] = Int[]
    end
    for i = 1:length(uf)
        k = find(cf, i)
        push!(district[k], i)
    end
    return district
end


"""
GreedyMerging merges districts greedily to repair the solution.

Parameters:
- `instance::Instance`: The problem instance
- `uf::UnionFind`: A UnionFind object representing the minimum spanning tree

Returns:
- `districts::Vector{Vector{Int}}`: A vector of vectors of nodes representing the districts
"""
function GreedyMerging(instance::Instance, uf::UnionFinder)
    targetNbDistricts = Int(floor(instance.num_blocks / instance.target_district_size))
    cf = CompressedFinder(uf)

    while cf.groups > targetNbDistricts
        districts = groupBlocks(uf)
        neighborPairs = findNeighborDistricts(instance, districts)
        minWeight = Inf
        mergePair = (0, 0)
        for (pair, weight) in neighborPairs
            if weight < minWeight
                minWeight = weight
                mergePair = pair
            end
        end

        mergeDistricts(uf, districts[mergePair[1]], districts[mergePair[2]])
        cf = CompressedFinder(uf)
    end

    districts = groupBlocks(uf)
    return [Vector{Int64}(subgraph) for subgraph in districts]
end

"""
    find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds the IDs of neighboring districts.

# Arguments
- `i::Int`: The district ID.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- A list of IDs for the neighboring districts.
"""

function find_id_neighbors(i::Int, neighborPairs::Dict{Tuple{Int,Int},Int})
    neighborIds = []
    for (pair, _) in neighborPairs
        if pair[1] == i
            push!(neighborIds, pair[2])
        elseif pair[2] == i
            push!(neighborIds, pair[1])
        end
    end
    return neighborIds
end

"""
    get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)

Retrieves nodes that are neighbors to a specific district.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `i::Int`: The index of the district to find neighbors for.

# Returns
- A list of nodes that are neighbors to the specified district.
"""

function get_nodes_neighbor(instance::Instance, districts::Vector{Vector{Int}}, i::Int)
    nodes = []
    for node in districts[i]
        for neighbor in neighbors(instance.graph, node)
            if neighbor in districts[i] ||
               neighbor == instance.num_blocks + 1 ||
               neighbor in nodes
                continue
            end
            push!(nodes, neighbor)
        end
    end
    return nodes
end

"""
    findSuitableNodeToAdd(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Finds a suitable node to add to a district that is under the minimum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing an additional node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to add and the district it belongs to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToAdd(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    #order the neighbor districts by decreasing size
    neighborIds = sort(neighborIds, by = x -> length(districts[x]), rev = true)
    neighbor_nodes = get_nodes_neighbor(instance, districts, district_index)
    # if no suitable node is found, try with the next district
    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        for node in districts[selected_district]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[selected_district], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) > instance.min_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    findSuitableNodeToRemove(instance::Instance, districts::Vector{Vector{Int}}, district_index::Int, neighborPairs::Dict{Tuple{Int, Int}, Int})

Identifies a suitable node to remove from a district that is over the maximum size limit.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts.
- `district_index::Int`: The index of the district needing to shed a node.
- `neighborPairs::Dict{Tuple{Int, Int}, Int}`: A dictionary mapping pairs of neighboring districts to their ID.

# Returns
- The node to remove and the new district it should belong to, or (0, 0) if no suitable node is found.
"""

function findSuitableNodeToRemove(
    instance::Instance,
    districts::Vector{Vector{Int}},
    district_index::Int,
    neighborPairs::Dict{Tuple{Int,Int},Int},
)
    neighborIds = find_id_neighbors(district_index, neighborPairs)
    neighborIds = sort(neighborIds, by = x -> length(districts[x]))

    for idx = 1:length(neighborIds)
        selected_district = neighborIds[idx]
        neighbor_nodes = get_nodes_neighbor(instance, districts, selected_district)
        for node in districts[district_index]
            if node in neighbor_nodes
                sub_graph, _ = induced_subgraph(
                    instance.graph,
                    setdiff(districts[district_index], [node]),
                )
                if is_connected(sub_graph) &&
                   length(districts[selected_district]) < instance.max_district_size
                    return node, selected_district
                end
            end
        end
    end
    return 0, 0
end

"""
    repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that do not meet the minimum size requirement by adding nodes from neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_min_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x))
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) < instance.min_district_size
            while length(districts[i]) < instance.min_district_size
                node, neighbor_district =
                    findSuitableNodeToAdd(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                push!(districts[i], node)
                districts[neighbor_district] = setdiff(districts[neighbor_district], [node])
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districts that exceed the maximum size limit by removing nodes to neighboring districts.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_max_districts(instance::Instance, districts::Vector{Vector{Int}})
    districts = sort(districts, by = x -> length(x), rev = true)
    neighborPairs = findNeighborDistricts(instance, districts)
    for i = 1:length(districts)
        if length(districts[i]) > instance.max_district_size
            while length(districts[i]) > instance.max_district_size
                node, new_district =
                    findSuitableNodeToRemove(instance, districts, i, neighborPairs)
                if node == 0
                    break
                end
                districts[i] = setdiff(districts[i], [node])
                push!(districts[new_district], node)
                neighborPairs = findNeighborDistricts(instance, districts)
            end
        end
    end
    return districts
end

"""
    repair_districts(instance::Instance, districts::Vector{Vector{Int}})

Repairs districting solutions that do not respect the district size constraints. It adjusts districts to meet both minimum and maximum size requirements.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be repaired.

# Returns
- The repaired list of districts.
"""

function repair_districts(instance::Instance, districts::Vector{Vector{Int}})
    MAX_ITER = 10
    iter = 1
    is_valid_solution = is_valid_districting_solution(instance, districts)
    while !is_valid_solution && iter <= MAX_ITER
        districts = repair_max_districts(instance, districts)
        districts = repair_min_districts(instance, districts)
        iter += 1
    end
    return districts
end


"""
    is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})

Checks whether all districts are valid, i.e., they are within the maximum and minimum district size constraints.

# Arguments
- `instance::Instance`: The problem instance.
- `districts::Vector{Vector{Int}}`: A list of districts to be checked.

# Returns
- `True` if all districts are valid, and `False` if any district is invalid.
"""

function is_valid_districting_solution(instance::Instance, districts::Vector{Vector{Int}})
    for i = 1:length(districts)
        districtSize = length(districts[i])
        isTooLarge = districtSize > instance.max_district_size
        isTooSmall = districtSize < instance.min_district_size
        if isTooLarge || isTooSmall
            return false
        end
    end
    return true
end


"""
    initialize_solution(instance::Instance, costloader::Costloader, mode::String="CMST", model = nothing)

Creates an initial solution for the problem instance. The solution respects district size constraints and minimizes the total cost.

# Arguments
- `instance::Instance`: The problem instance.
- `costloader::Costloader`: The costloader object.
- `mode::String`: The mode of the solution (e.g., "CMST" or "Districting", etc.)

# Returns
- `solution::Solution`: The initial solution.
"""

function initialize_solution(
    instance::Instance,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    tree = run_kruskal(instance)
    districts = GreedyMerging(instance, tree)
    districts = repair_districts(instance, districts)
    return create_solution(instance, districts, costloader, mode, model)
end

"""
    is_valid_move(instance::Instance, solution::Solution, node::Int, current_district::District, target_district::District)::Bool

Checks if moving a node from its current district to a target district is a valid move according to the constraints.

# Arguments
- `instance::Instance`: The problem instance.
- `solution::Solution`: The current solution.
- `node::Int`: The node to be moved.
- `current_district::District`: The district from which the node is being moved.
- `target_district::District`: The district to which the node is being moved.

# Returns
- `Bool`: `true` if the move is valid, `false` otherwise.
"""

function is_valid_move(
    instance::Instance,
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
)::Bool
    if node in target_district.nodes
        return false
    end

    if length(target_district.nodes) == instance.max_district_size
        return false
    end

    if !is_subgraph_connected(instance.graph, [node; target_district.nodes])
        return false
    end

    if length(current_district.nodes) > 1 &&
       !is_subgraph_connected(instance.graph, setdiff(current_district.nodes, [node]))
        return false
    end

    desired_number =
        floor(solution.instance.num_blocks / solution.instance.target_district_size)

    if solution.nb_districts <= desired_number &&
       length(current_district.nodes) == instance.min_district_size
        return false
    end

    if length(current_district.nodes) == 1 &&
       length(solution.districts) - 1 < desired_number
        return false
    end

    return true
end

"""
    apply_move!(solution::Solution, node::Int, current_district::District, target_district::District, costloader::Costloader, mode::String, model = nothing)

Applies a move by transferring a node from its current district to a target district, updating the districts and solution costs accordingly.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be moved.
- `current_district::District`: The district from which the node is being moved.
- `target_district::District`: The district to which the node is being moved.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Modifies the solution by applying the move and updating costs.
"""

function apply_move!(
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
    costloader::Costloader,
    mode::String,
    model = nothing,
)
    deleteat!(current_district.nodes, findfirst(x -> x == node, current_district.nodes))
    push!(target_district.nodes, node)
    current_district.cost =
        compute_district_cost(solution.instance, current_district, costloader, mode, model)
    target_district.cost =
        compute_district_cost(solution.instance, target_district, costloader, mode, model)
    solution.cost = compute_solution_cost(solution)
end

"""
    revert_move!(solution::Solution, node::Int, current_district::District, target_district::District, idCurrentDistrict::Int, costloader::Costloader, mode::String, model = nothing)

Reverts a previously applied move, restoring the node to its original district.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be reverted.
- `current_district::District`: The district to which the node is being reverted.
- `target_district::District`: The district from which the node is being moved back.
- `idCurrentDistrict::Int`: The ID of the current district.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Modifies the solution by reverting the move and updating costs.
"""

function revert_move!(
    solution::Solution,
    node::Int,
    current_district::District,
    target_district::District,
    idCurrentDistrict::Int,
    costloader::Costloader,
    mode::String,
    model = nothing,
)
    deleteat!(target_district.nodes, findfirst(x -> x == node, target_district.nodes))
    push!(current_district.nodes, node)

    current_district.cost =
        compute_district_cost(solution.instance, current_district, costloader, mode, model)
    target_district.cost =
        compute_district_cost(solution.instance, target_district, costloader, mode, model)
    solution.cost = compute_solution_cost(solution)
    solution.blocks_district_ids[node] = idCurrentDistrict
end

"""
    relocate(solution::Solution, node::Int, idTargetDistrict::Int, costloader::Costloader, mode::String="CMST", PERTURBATION_PROBABILITY::Bool=false, model = nothing)

Relocates a node from its current district to a target district if the move is valid.

# Arguments
- `solution::Solution`: The current solution.
- `node::Int`: The node to be relocated.
- `idTargetDistrict::Int`: The ID of the target district.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `PERTURBATION_PROBABILITY::Bool`: The probability of applying the perturbation.
- `model`: An optional model used for cost computation.

# Returns
- `Bool`: `true` if the relocation is successful, `false` otherwise.
"""

function relocate(
    solution::Solution,
    node::Int,
    idTargetDistrict::Int,
    costloader::Costloader,
    mode::String = "CMST",
    PERTURBATION_PROBABILITY::Bool = false,
    model = nothing,
)
    idCurrentDistrict = solution.blocks_district_ids[node]
    current_district = solution.districts[idCurrentDistrict]
    target_district = solution.districts[idTargetDistrict]

    return if !is_valid_move(
        solution.instance,
        solution,
        node,
        current_district,
        target_district,
    )
        false
    else
        old_cost = solution.cost
        apply_move!(
            solution,
            node,
            current_district,
            target_district,
            costloader,
            mode,
            model,
        )
        if solution.cost >= old_cost && !PERTURBATION_PROBABILITY
            revert_move!(
                solution,
                node,
                current_district,
                target_district,
                idCurrentDistrict,
                costloader,
                mode,
                model,
            )
            false
        else
            solution.blocks_district_ids[node] = idTargetDistrict
            if isempty(current_district.nodes)
                deleteat!(solution.districts, idCurrentDistrict)
                for i in eachindex(solution.blocks_district_ids)
                    if solution.blocks_district_ids[i] > idCurrentDistrict
                        solution.blocks_district_ids[i] -= 1
                    end
                end
            end
            solution.nb_districts = length(solution.districts)
            solution.is_feasible = is_solution_feasible(solution)
            true
        end
    end
end

"""
    swap(solution::Solution, node1::Int, node2::Int, costloader::Costloader, mode::String="CMST", PERTURBATION_PROBABILITY::Bool=false, model = nothing)

Swaps two nodes between their respective districts if the move is valid.

# Arguments
- `solution::Solution`: The current solution.
- `node1::Int`: The first node to be swapped.
- `node2::Int`: The second node to be swapped.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `PERTURBATION_PROBABILITY::Bool`: The probability of applying the perturbation.
- `model`: An optional model used for cost computation.

# Returns
- `Bool`: `true` if the swap is successful, `false` otherwise.
"""

function swap(
    solution::Solution,
    node1::Int,
    node2::Int,
    costloader::Costloader,
    mode::String = "CMST",
    PERTURBATION_PROBABILITY::Bool = false,
    model = nothing,
)
    # get the instance
    instance = solution.instance
    # get the current district of the node
    idCurrentDistrict1 = solution.blocks_district_ids[node1]
    idCurrentDistrict2 = solution.blocks_district_ids[node2]
    # get the current district
    current_district1 = solution.districts[idCurrentDistrict1]
    current_district2 = solution.districts[idCurrentDistrict2]
    # check if the nodes are in the same district
    if idCurrentDistrict1 == idCurrentDistrict2
        return false
    end
    # Check if the districts remain connected after the swap
    if !(
        is_subgraph_connected(
            instance.graph,
            [node2; setdiff(current_district1.nodes, [node1])],
        ) && is_subgraph_connected(
            instance.graph,
            [node1; setdiff(current_district2.nodes, [node2])],
        )
    )
        #println("Districts are not connected after swap.")
        return false
    end
    # get a copy of the cost of the solution before the relocation
    old_cost = deepcopy(solution.cost)
    # remove the node from the current district
    deleteat!(current_district1.nodes, findall(x -> x == node1, current_district1.nodes))
    deleteat!(current_district2.nodes, findall(x -> x == node2, current_district2.nodes))

    # add the node to the target district
    push!(current_district1.nodes, node2)
    push!(current_district2.nodes, node1)
    # update the feasibility of the districts
    current_district1.is_feasible = is_district_feasible(instance, current_district1)
    current_district2.is_feasible = is_district_feasible(instance, current_district2)
    # update the cost of the districts
    current_district1.cost =
        compute_district_cost(instance, current_district1, costloader, mode, model)
    current_district2.cost =
        compute_district_cost(instance, current_district2, costloader, mode, model)

    # update the cost of the solution
    solution.cost = compute_solution_cost(solution)
    #check if the cost of the solution has improved
    if (solution.cost >= old_cost && !(PERTURBATION_PROBABILITY))
        # revert the relocation
        # remove the node from the target district
        deleteat!(
            current_district1.nodes,
            findall(x -> x == node2, current_district1.nodes),
        )
        deleteat!(
            current_district2.nodes,
            findall(x -> x == node1, current_district2.nodes),
        )

        # add the node to the current district
        push!(current_district1.nodes, node1)
        push!(current_district2.nodes, node2)

        # update the feasibility of the districts
        current_district1.is_feasible = is_district_feasible(instance, current_district1)
        current_district2.is_feasible = is_district_feasible(instance, current_district2)
        # update the cost of the districts
        current_district1.cost =
            compute_district_cost(instance, current_district1, costloader, mode, model)
        current_district2.cost =
            compute_district_cost(instance, current_district2, costloader, mode, model)
        # update the cost of the solution
        solution.cost = compute_solution_cost(solution)
        # update the district id of the node
        solution.blocks_district_ids[node1] = idCurrentDistrict1
        solution.blocks_district_ids[node2] = idCurrentDistrict2
        # update the number of districts
        solution.nb_districts = length(solution.districts)
        # update the feasibility of the solution
        solution.is_feasible = is_solution_feasible(solution)
        return false

    else
        # update the district id of the node
        solution.blocks_district_ids[node1] = idCurrentDistrict2
        solution.blocks_district_ids[node2] = idCurrentDistrict1
        # update the number of districts
        solution.nb_districts = length(solution.districts)
        # update the feasibility of the solution
        solution.is_feasible = is_solution_feasible(solution)
        return true
    end

end


"""
    localSearch(init_solution::Solution, costloader::Costloader, start_time, mode::String="CMST", model = nothing)

Conducts local search  on a solution, applying relocation and swap moves to improve the solution.

# Arguments
- `init_solution::Solution`: The initial solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `start_time`: The start time of the search for time tracking.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Returns
- The improved solution after a single local search iteration.
"""

function localSearch(
    init_solution::Solution,
    costloader::Costloader,
    start_time::Float64,
    mode::String = "CMST",
    model = nothing,
)

    change = true
    currentSolution = deepcopy(init_solution)
    while change
        if (time() - start_time) > MAX_TIME
            return currentSolution
        end
        change = false
        random_district_order = randperm(currentSolution.nb_districts)
        for i in random_district_order
            for j in random_district_order
                if (time() - start_time) > MAX_TIME
                    return currentSolution
                end
                if (i <= j)
                    continue
                end
                best_move = Dict()
                best_cost = currentSolution.cost
                border_nodes = get_border_nodes(currentSolution, i, j)
                if length(border_nodes) > 0
                    border1, border2 = border_nodes
                    for node1 in border1
                        if (time() - start_time) > MAX_TIME
                            break
                        end
                        solution = deepcopy(currentSolution)
                        relocate(solution, node1, j, costloader, mode, false, model)
                        if solution.cost < best_cost
                            best_move =
                                Dict(:node => node1, :district => j, :type => "relocate")
                            best_cost = solution.cost
                        end
                        for node2 in border2
                            if (time() - start_time) > MAX_TIME
                                break
                            end
                            solution = deepcopy(currentSolution)
                            swap(solution, node1, node2, costloader, mode, false, model)
                            if solution.cost < best_cost
                                best_move =
                                    Dict(:node1 => node1, :node2 => node2, :type => "swap")
                                best_cost = solution.cost
                            end
                        end
                    end
                    for node2 in border2
                        if (time() - start_time) > MAX_TIME
                            break
                        end
                        solution = deepcopy(currentSolution)
                        relocate(solution, node2, i, costloader, mode, false, model)
                        if solution.cost < best_cost
                            best_move =
                                Dict(:node => node2, :district => i, :type => "relocate")
                            best_cost = solution.cost
                        end
                    end
                end
                if best_cost < currentSolution.cost
                    if best_move[:type] == "relocate"
                        relocate(
                            currentSolution,
                            best_move[:node],
                            best_move[:district],
                            costloader,
                            mode,
                            false,
                            model,
                        )
                    elseif best_move[:type] == "swap"
                        swap(
                            currentSolution,
                            best_move[:node1],
                            best_move[:node2],
                            costloader,
                            mode,
                            false,
                            model,
                        )
                    end
                    change = true
                end
            end
        end
    end
    return currentSolution
end

"""
    perturbation(solution::Solution, costloader::Costloader, mode::String="CMST", model = nothing)

Applies perturbation moves to a solution to escape local optima. It randomly relocates and swaps nodes between districts.

# Arguments
- `solution::Solution`: The current solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Effects
Randomly modifies the solution to explore new configurations.
"""

function perturbation(
    solution::Solution,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    # Create a random permutation of district indices
    random_district_order = randperm(solution.nb_districts)
    for i in random_district_order
        # Create a random permutation of district indices for the inner loop
        random_district_order_inner = randperm(solution.nb_districts)
        for j in random_district_order_inner
            if (i <= j)
                continue
            end
            border_nodes = get_border_nodes(solution, i, j)
            if length(border_nodes) > 0
                border1, border2 = border_nodes
                @views for node1 in border1
                    if (rand() > PERTURBATION_PROBABILITY)
                        relocate(solution, node1, j, costloader, mode, true, model)
                    end
                    @views for node2 in border2
                        if (rand() > PERTURBATION_PROBABILITY)
                            swap(solution, node1, node2, costloader, mode, true, model)
                        end
                    end
                end
                @views for node2 in border2
                    if (rand() > PERTURBATION_PROBABILITY)
                        relocate(solution, node2, i, costloader, mode, true, model)
                    end
                end
            end
        end
    end
end

"""
    Iterated_local_search(solution::Solution, costloader::Costloader, mode::String="CMST", model = nothing)

Implements the Iterated Local Search (ILS) algorithm to improve a given solution. It combines local search with perturbations.

# Arguments
- `solution::Solution`: The initial solution.
- `costloader::Costloader`: The loader containing precomputed district costs.
- `mode::String`: The mode of operation.
- `model`: An optional model used for cost computation.

# Returns
- The improved solution after applying the ILS algorithm.
"""

function Iterated_local_search(
    solution::Solution,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    best_solution = deepcopy(solution)


    start_time = time()
    iters = 0
    solution = localSearch(solution, costloader, start_time, mode, model)
    while (time() - start_time) < MAX_TIME
        perturbation(solution, costloader, mode, model)
        solution = localSearch(solution, costloader, start_time, mode, model)
        if solution.cost < best_solution.cost
            best_solution = deepcopy(solution)
            iters = 0
        else
            iters += 1
        end
    end
    return best_solution
end

"""
    ILS_solve_instance(instance, costloader, solver_type, params)

Solves an instance of the districting problem using an Iterated Local Search (ILS) algorithm. It repeatedly tries to find a feasible solution by updating edge weights and applying ILS.

# Arguments
- `instance`: The problem instance.
- `costloader`: The costloader containing precomputed costs.
- `solver_type`: The type of solver to use (e.g., "CMST").
- `params`: Parameters for the solver.

# Returns
- `solution`: The solution found by the ILS algorithm.
"""

function ILS_solve_instance(instance, costloader, solver_type, params)
    update_edge_weights!(
        instance.graph,
        solver_type == "CMST" ? params : rand(ne(instance.graph)),
    )
    solution = initialize_solution(instance, costloader, solver_type, params)
    println("Initial solution feasibility before while loop: ", solution.is_feasible)
    max_try = 10
    while !solution.is_feasible && solver_type != "CMST" && max_try > 0
        W = rand(ne(instance.graph))
        update_edge_weights!(instance.graph, W)
        solution = initialize_solution(instance, costloader, solver_type, params)
    end
    println(
        "Initial solution feasibility after while loop: ",
        solution.is_feasible,
        " with cost: ",
        solution.cost,
    )
    println("max try: ", max_try)
    solution = Iterated_local_search(solution, costloader, solver_type, params)
    return solution
end
