using InferOpt
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2

include("instance.jl")
include("ils.jl")

const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_DATA = 200
const NB_BU_SMALL = 30
const TARGET_DISTRICT_SIZE = 3
const DEFAULT_CITY = "Manchester" # change this to perform different experiments!

function predict_theta(instance, model, tmean, tstd)
    println("Getting features...")
    features = get_instance_features(instance)
    features = (features .- tmean) ./ tstd
    println("Constructing input...")
    input = create_edge_graph(instance.graph, features)
    println("Finding theta...")
    theta = model(input)
    theta = reshape(-theta, :,)
    return theta
end

function find_solution_city(model, mean_train, std_train)
    city = DEFAULT_CITY
    depot_location = DEPOT_LOCATION
    NB_BU = NB_BU_LARGE
    target_district_size = TARGET_DISTRICT_SIZE
    instance = build_instance(city, NB_BU, target_district_size, depot_location)
    W = rand(ne(instance.graph))
    update_edge_weights!(instance.graph, W)
    println("obtaining theta")
    theta = predict_theta(instance, model, mean_train, std_train)
    println("theta achieved")
    update_edge_weights!(instance.graph, theta)
    costloader = Costloader([], [])
    solution = ILS_solve_instance(instance, costloader, "CMST", theta)
    return solution
end

function export_solution(solution, filename)
    for district in solution.districts
        district.nodes = sort(district.nodes)
    end

    file = open(filename, "w")
    write(file, "$(solution.instance.city_name)\n")
    write(file, "$(solution.instance.num_blocks)\n")
    write(file, "$(solution.nb_districts)\n")
    write(file, "COST $(solution.cost)\n")
    write(file, "$(solution.is_feasible)\n")
    for district in solution.districts
        for node in district.nodes
            write(file, "$node ")
        end
        write(file, "\n")
    end
    close(file)
    println("Solution exported to $filename")
end
