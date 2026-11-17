# condensed version of train and fit from DistrictNet. parameter nb = 100, model = districtNet257

export train_and_fit

include("../dependencies/julia/instance.jl")
include("gnn.jl")

# these hyperparameters are taken from original DistrictNet.

# Training Hyperparameters
const SAMPLES = 20
const SAMPLES_MONTE_CARLO = 20
const BATCH_SIZE = 1
const EPOCHS = 100
const EPSILON = 1.0
const LR = 0.001

# Solver Hyperparameters
const MAX_TIME = 600 # was originally 1200 = 20 mins. 
const PERTURBATION_PROBABILITY = 0.985

# Constants definition
const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_DATA = 200
const NB_BU_SMALL = 30

# main train/fit function

function train_and_fit()
"""
return model to be fitted
"""
    hyperparameters = (SAMPLES, SAMPLES_MONTE_CARLO, BATCH_SIZE, HIDDEN_SIZE, EPOCHS, EPSILON, LR)
    nb_data = NB_TRAIN
    data_train = aggregate_city_training_data(nb_data)
    data_test = aggregate_city_training_data(NB_TEST, nb_data + 1)
    println("Training data size: ", length(data_train), ". Test data size: ", length(data_test))
    data_train, mean_train, stdev_train = normalize_features(data_train)
    data_test, _, _ = normalize_features(data_test, mean_train, stdev_train)
    model_name = "models/DistrictNet257_$(nb_data).jld2"
    initial_graph = data_train[1][1].instance.graph
    input_size = length(props(initial_graph, 1))-1
    if isfile(model_name)
        @info "Loading model from file..."
        model_state = JLD2.load(model_name, "model_state")
        model = build_gnn_model(input_size, HIDDEN_SIZE)
        Flux.loadmodel!(model, model_state)
        return model, mean_train, stdev_train
    end
    model = build_gnn_model(input_size, HIDDEN_SIZE)
    model = train_gnn_model(model, data_train, data_test, hyperparameters)
    model_state = Flux.state(model)
    if !isdir("models")
        mkpath("models")
    end
    jldsave(model_name; model_state, mean_train, stdev_train)
    return model, mean_train, stdev_train
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_and_fit()
end
