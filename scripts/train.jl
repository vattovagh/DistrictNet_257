# Usage: julia --project scripts/train.jl <experiment> <model> <district_size> <n_units>
# TODO: load data, build model, train, save weights to results/

# condensed version of train and fit from DistrictNet. parameter nb = 100, model = districtNet257
# TODO: implement other functions called in this one. They currently aren't in this project, and this code won't run yet.

export train_and_fit

include("../src/instance.jl")

# these hyperparameters are taken from original DistrictNet.

# Constants definition
const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_DATA = 200
const NB_BU_SMALL = 30
# Training Hyperparameters
const BATCH_SIZE = 1
const NB_SAMPLES = 20
const EPSILON = 1.0
const WEIGHT_DECAY = 0.0
const INITIAL_LR = 0.001
const NB_EPOCHS = 100
const MC_SAMPLES = 20

# Solver Hyperparameters
const MAX_TIME = 600 # was originally 1200 = 20 mins. 
const PERTURBATION_PROBABILITY = 0.985

# helpers

function decayed_learning_rate(epoch, initial_lr,
                               decay_steps=10, decay_rate=0.9, min_lr=0.0001)
    lr = max(initial_lr * (decay_rate ^ floor(epoch / decay_steps)), min_lr)
    return lr
end

function randomize_data(data, mc_samples=1)
    temps_data = []
    for item in data
        y = randomized_constructor(item[1].solution, mc_samples)
        push!(temps_data, (item[1], y))
    end
    return temps_data
end

# main train/fit function

function train_and_fit()
"""
return model to be fitted
"""
    nb_data = NB_TRAIN
    data_train = aggregate_city_training_data(nb_data)
    data_test = aggregate_city_training_data(NB_TEST, nb_data + 1)
    println("Training data size: ", length(data_train), ", test data size: ", length(data_test))
    data_train, mean_train, std_train = normalize_features(data_train)
    data_test, _, _ = normalize_features(data_test, mean_train, std_train)
    hyperparams = (BATCH_SIZE, HIDDEN_SIZE, NB_SAMPLES, EPSILON, WEIGHT_DECAY, INITIAL_LR, NB_EPOCHS, MC_SAMPLES)
    model_name = "models/DistrictNet257_$(nb_data).jld2"
    initial_graph = data_train[1][1].instance.graph
    input_size = length(props(initial_graph, 1))-1
    if isfile(model_name)
        @info "Loading model from file..."
        model_state = JLD2.load(model_name, "model_state")
        model = build_gnn_model(input_size, HIDDEN_SIZE)
        Flux.loadmodel!(model, model_state)
        return model, mean_train, std_train
    end
    model = build_gnn_model(input_size, HIDDEN_SIZE)
    model = train_GNN(model, data_train, data_test, hyperparams)
    model_state = Flux.state(model)
    if !isdir("models")
        mkpath("models")
    end
    jldsave(model_name; model_state, mean_train, std_train)
    return model, mean_train, std_train
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_and_fit()
end
