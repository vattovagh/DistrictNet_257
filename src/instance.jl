# Core types shared by every other file.

# TODO: BasicUnit  -- one geographic unit (id, centroid, population/demand, area, ...)
# TODO: City       -- units + adjacency Graph + metadata (name, depot location)
# TODO: District   -- set of unit ids, must be connected and within capacity
# TODO: Partition  -- vector of Districts covering all units exactly once
# TODO: validate(partition, city) -> Bool  (contiguity, capacity, coverage)

using InferOpt
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2

include("structs.jl")
include("solution.jl")
include("solver.jl")
include("district.jl")
include("utils.jl")
include("../scripts/gnn.jl")
include("evaluation.jl")

Random.seed!(1234)

const NB_BU_SMALL = 30
const DEPOT_LOCATION = "C"
const MC_SAMPLES = 20
const NB_SCENARIO = 100
const PENALTY = 10000
const NB_TRAIN = 100
const NB_TEST = 20
const HIDDEN_SIZE = 64

# helpers

function get_depot_location(
    corners::Tuple{Float64,Float64,Float64,Float64},
    depot_location::String,
)::Tuple{Float64,Float64}
    min_lat, min_lon, max_lat, max_lon = corners
    if depot_location == "C"
        return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    elseif depot_location == "NW"
        return (min_lon, max_lat)
    elseif depot_location == "NE"
        return (max_lon, max_lat)
    elseif depot_location == "SW"
        return (min_lon, min_lat)
    elseif depot_location == "SE"
        return (max_lon, min_lat)
    else
        error("Invalid depot location: $depot_location")
    end
end

function parse_json(file_path::String)::Dict
    try
        return JSON.parsefile(file_path; dicttype=Dict{String, Any})
    catch e
        error("Failed to parse file at $file_path. Error: $e")
    end
end

function is_subgraph_connected(graph::MetaGraph, nodes::Vector{Int})::Bool
    subgraph, _ = induced_subgraph(graph, nodes)
    return is_connected(subgraph)
end

function extract_edge_feature(g::MetaGraph)
    function get_filtered_properties(dict)
        return [val for (key, val) in dict if key ∉ [:id, :centre]]
    end
    edge_data = [
        begin
            prop_src, prop_dst = props(g, src(e)), props(g, dst(e))
            vector_src, vector_dst =
                get_filtered_properties(prop_src), get_filtered_properties(prop_dst)
            dist = euclidean_distance(prop_src[:centre], prop_dst[:centre])
            vcat(0.5 * (vector_src + vector_dst), dist)
        end for e in edges(g)
    ]

    return [Vector{Float32}(feature) for feature in edge_data]
end

function randomized_constructor(solution, mc_samples)
    # return y as the monte carlo solution
    y = zeros(ne(solution.instance.graph))
    for i in 1:mc_samples
        W = rand(ne(solution.instance.graph))
        update_edge_weights!(solution.instance.graph, W)
        y_rand = get_solution_edges(solution)
        y += y_rand
    end
    # average the solution
    y = y/mc_samples
    return y
end

function set_block_properties!(G::MetaGraph, props::Dict, idx::Int, jsondata::Dict, depot_corr::Array{Float64,1})
"""
G = graph where properties are added
props = dictionary of properties for block
idx = index of block in graph
jsondata = additional data for property calculation
"""
    #longlat_ref = jsondata["metadata"]["REFERENCE_LONGLAT"]
    centre = props["centre"]
    distance = norm(depot_corr - centre) * 1000
    compactness = calculate_polsby_popper_score(props["AREA"], props["PERIMETER"])
    sw_compact = calculate_schwartzberg_compactness(props["AREA"], props["PERIMETER"])

    block_properties = Dict(
        :id => props["ID"] + 1,
        :population => props["POPULATION"],
        :area => props["AREA"],
        :density => props["DENSITY"],
        :perimeter => props["PERIMETER"],
        :compactness => compactness,
        :centre => centre,
        :distDepot => distance,
    )

    set_props!(G, idx, block_properties)
end

function update_edge_weights!(graph::MetaGraph, weights::Vector{T}) where {T<:Real}
    if length(weights) != ne(graph)
        error("$(length(weights)) weights, but $(ne(graph)) edges provided")
    end
    for (i, e) in enumerate(edges(graph))
        set_prop!(graph, src(e), dst(e), :weight, weights[i])
    end
end

function add_edges!(G::MetaGraph, props::Dict, idx::Int, num_blocks::Int)
    for adjacent in props["LIST_ADJACENT"]
        adjacent <= num_blocks && add_edge!(G, idx, adjacent + 1)
    end
end

function calculate_polsby_popper_score(area::Float64, perimeter::Float64)::Float64
    return 4 * π * area / perimeter^2
end

function calculate_schwartzberg_compactness(area::Float64, perimeter::Float64)::Float64
    return perimeter / sqrt(4 * π * area)
end

function calculate_bounding_box(features)
    min_lat, min_lon = Inf, Inf
    max_lat, max_lon = -Inf, -Inf

    for feature in features
        for polygon in feature["geometry"]["coordinates"]
            # check if polygon is multi-polygon
            if length(polygon) == 1
                polygon = polygon[1]
            end
            for point in polygon
                lon, lat = point
                min_lat = min(min_lat, lat)
                min_lon = min(min_lon, lon)
                max_lat = max(max_lat, lat)
                max_lon = max(max_lon, lon)
            end
        end
    end

    return (min_lat, min_lon, max_lat, max_lon)
end

# main helper

function build_instance(
    city_name::String,
    num_blocks::Int,
    target_district_size::Int,
    depot_location::String,
    )::Instance

    # create city graph from DistrictNet

    G = MetaGraph(num_blocks + 1)
    file_path = joinpath("data/cities", "$city_name.geojson")
    city_data = parse_json(file_path)
    blocks_features = city_data["features"][1:num_blocks]
    corners = calculate_bounding_box(blocks_features)
    depot_corr = collect(get_depot_location(corners, depot_location))    
    for i = 1:num_blocks
        block_props = city_data["features"][i]["properties"]
        # println(typeof(block_props))
        set_block_properties!(G, block_props, i, city_data, depot_corr)
        add_edges!(G, block_props, i, num_blocks)
    end

    depot_properties = Dict(
        :id => num_blocks + 1,
        :population => 0,
        :area => 0,
        :density => 0,
        :perimeter => 0,
        :compactness => 0,
        :centre => depot_corr,
        :distDepot => 0,
    )
    set_props!(G, num_blocks + 1, depot_properties)
    foreach(i -> add_edge!(G, num_blocks + 1, i), 1:num_blocks)
    graph = G

    # build instance from DistrictNet

    min_size = floor(Int, 0.8 * target_district_size)
    max_size = ceil(Int, 1.2 * target_district_size)
    return Instance(
        city_name,
        num_blocks,
        target_district_size,
        min_size,
        max_size,
        graph,
        depot_location,
    )
end

# main function

function aggregate_city_training_data(nb_data, start=1)
    TARGET_DISTRICT_SIZE = 3
    data_train = []
    cities = ["city" * string(i) for i = start:nb_data+start-1]
    for CITY in cities
        instance = build_instance(CITY, NB_BU_SMALL, TARGET_DISTRICT_SIZE, DEPOT_LOCATION)
        W = W = rand(ne(instance.graph))
        update_edge_weights!(instance.graph, W)
        path = string("data/costs/", CITY, "_", DEPOT_LOCATION, "_", NB_BU_SMALL, "_", TARGET_DISTRICT_SIZE, "_tsp.train_and_test.json")
        costloader = load_precomputed_costs(path)
        solution, unique_subgraphs = districting_exact_solver(instance, costloader)
        if (solution == nothing)
            continue
        end
        cost = solution.cost
        y = randomized_constructor(solution, MC_SAMPLES)
        g = deepcopy(instance.graph)
        edges_data = extract_edge_feature(g)
        data = hcat(edges_data...)
        gnn_graph = create_edge_graph(g, data)
        push!(
            data_train,
            (
                GraphFeaturesInstance(
                    data,
                    instance,
                    cost,
                    gnn_graph,
                    unique_subgraphs,
                    costloader,
                    solution,
                ),
                y,
            ),
        )
    end
    return data_train
end

if abspath(PROGRAM_FILE) == @__FILE__
    aggregate_city_training_data(100)
end
