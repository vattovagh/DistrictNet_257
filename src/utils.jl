# takes file string to load precomputed costs from, returns Costloader (district lists, district costs)

function load_precomputed_costs(path::String)
    data = open(path) do file
        JSON.parse(file)
    end
    raw_districts = data["districts"]
    districts_lists = []
    district_costs = []
    for i = 1:length(raw_districts)
        block_list = raw_districts[i]["list-blocks"] .+ 1
        push!(districts_lists, block_list)
        push!(district_costs, raw_districts[i]["average-cost"])
    end
    districts_lists = [Vector{Int64}(d) for d in districts_lists]
    district_costs = Float64.(district_costs)
    costloader = Costloader(districts_lists, district_costs)
    #clear memory
    data = nothing
    raw_districts = nothing
    districts_lists = nothing
    district_costs = nothing

    return costloader
end

function check_1d_in_2d(lst1d, lst2d)
    return findfirst(vec -> are_vectors_equal(lst1d, vec), lst2d)
end

function create_edge_graph(g, node_features)
    # Convert node features to Float32
    node_features_float32 = Float32.(node_features)
    # Compute edge graph adjacency matrix
    nb_edges = ne(g)
    adj = falses(nb_edges, nb_edges)

    for edge in edges(g)
        u1, u2 = src(edge), dst(edge)
        idx = get_edge_id(g, u1, u2)

        for u in (u1, u2)
            neighbors_u = neighbors(g, u)
            idx2 = filter(x -> x != idx, [get_edge_id(g, u, v) for v in neighbors_u]) # Filter out self-loops
            adj[idx, idx2] .= true
            adj[idx2, idx] .= true
        end
    end
    # Create edge graph and GNN graph objects
    edge_graph = SimpleGraph(adj)
    return GNNGraph(edge_graph; ndata = node_features_float32)
end

function are_vectors_equal(vec1::Vector{Int}, vec2::Vector{Int})::Bool
    return length(vec1) == length(vec2) && all(sort(vec1) .== sort(vec2))
end

function euclidean_distance(p1, p2)
    return sqrt(sum((p1[i] - p2[i])^2 for i = 1:length(p1))) * 1000
end

function get_edge_id(g, u, v)
    g_edges = [(src(i), dst(i)) for i in edges(g)]
    idx = findfirst(x -> x == (u, v) || x == (v, u), g_edges)
    return idx
end
