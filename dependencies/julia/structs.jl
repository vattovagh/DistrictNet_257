struct Edge
    src::Int
    dst::Int
    weight::Float64
end

mutable struct Instance
    city_name::String
    num_blocks::Int
    target_district_size::Int
    min_district_size::Int
    max_district_size::Int
    graph::MetaGraph
    depot::String
end

mutable struct Costloader
    DistrictIds::Vector{Vector{Int}}
    Cost::Array{Float64}
end

mutable struct District
    id::Int
    nodes::Array{Int}
    cost::Float64
    is_feasible::Bool
end

mutable struct Solution
    instance::Instance
    districts::Array{District}
    cost::Float64
    nb_districts::Int
    is_feasible::Bool
    blocks_district_ids::Array{Int}
end


mutable struct GraphFeaturesInstance
    feature::Array{Real}
    instance::Instance
    cost::Float64
    gnn_graph::GNNGraph
    subgraphs::Vector{Vector{Int}}
    costloader::Costloader
    solution::Solution
end
