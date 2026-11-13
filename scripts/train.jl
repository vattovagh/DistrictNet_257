# Usage: julia --project scripts/train.jl <experiment> <model> <district_size> <n_units>
# TODO: load data, build model, train, save weights to results/

# condensed version of train and fit from DistrictNet. parameter nb = 100, model = districtNet257
# TODO: implement other functions called in this one. They currently aren't in this project, and this code won't run yet.

# these hyperparameters are taken from original DistrictNet.

export train_and_fit
using InferOpt, Plots
using MLUtils, GraphNeuralNetworks, Graphs, MetaGraphs, GraphPlot
using JSON, FilePaths, Random, LinearAlgebra
using Distributions, Statistics, UnionFind, Flux
using DataStructures, Serialization, FileIO
using Base.Filesystem: isfile
using Combinatorics, Optim
using JuMP, GLPK, CxxWrap, DistributedArrays, JLD2

# Constants definition
const NB_BU_LARGE = 120
const DEPOT_LOCATION = "C"
const NB_DATA = 200
const NB_BU_SMALL = 30
const NB_SCENARIO = 100
# Training Hyperparameters
const BATCH_SIZE = 1
const HIDDEN_SIZE = 64
const NB_SAMPLES = 20
const EPSILON = 1.0
const WEIGHT_DECAY = 0.0
const INITIAL_LR = 0.001
const NB_EPOCHS = 100
const MC_SAMPLES = 20

# Solver Hyperparameters
const MAX_TIME = 1200
const PERTURBATION_PROBABILITY = 0.985
const PENALTY = 10000

function build_gnn_model(input_size, hidden_size)
"""
# Args
- input_size: size of graph that the model will be used for
- hidden_size: size for hidden layers
# Return
- GNN model
"""
    model = Chain(
        GNNChain(
            GraphConv(input_size => hidden_size, relu; aggr=mean), 
            GraphConv(hidden_size => hidden_size, relu; aggr=mean),
            GraphConv(hidden_size => hidden_size; aggr=mean),
            Dense(hidden_size => hidden_size, relu),
            Dense(hidden_size => hidden_size),
            Dense(hidden_size => 1),
            vec,
        ),    x -> x.ndata.x,
        )
    return model
end

function train_GNN(model, data_train, data_test, hyperparams)
    start_time = time()

    batch_size, hidden_size, nb_samples, epsilon, weight_decay, initial_lr, nb_epochs, mc_samples = hyperparams
    regularized_predictor = PerturbedAdditive(cmst_exact_solver; ε=epsilon, nb_samples=nb_samples, seed=1234, is_parallel=true)
    loss = FenchelYoungLoss(regularized_predictor)

    # Loss function for Flux
    flux_loss(x, y) = loss(model(x.gnn_graph), y; instance=x.instance, subgraphs=x.subgraphs) 
    # Compute total loss for a batch
    flux_loss_batch(batch) = sum(flux_loss(item[1], item[2]) for item in batch) 

    data_train = randomize_data(data_train, mc_samples)
    train_dataset = Flux.DataLoader(data_train, batchsize=batch_size, shuffle=true)
    # DataLoader and optimizer
    opt = Flux.Optimise.Adam(initial_lr)
    LinearAlgebra.BLAS.set_num_threads(10)
    for epoch in 1:nb_epochs
        Flux.trainmode!(model, true)
        opt.eta = decayed_learning_rate(epoch, initial_lr)
        # Training
        train_loss = 0
        for batch in train_dataset
            batch_loss = 0
            grads = gradient(Flux.params(model)) do
                batch_loss  = flux_loss_batch(batch)
            end
            train_loss += batch_loss 
            Flux.update!(opt, Flux.params(model), grads)
        end

        # Normalize training loss
        train_loss /= length(data_train)
        println("Epoch $epoch: train loss = $train_loss")
    end
    end_time = time()

    @info "Training done. It took $(end_time - start_time) seconds."
    return model
end

function train_and_fit()
"""
return model to be fitted
"""
    nb_data = 100
    data_train = aggregateCityTrainingData(nb_data)
    data_train, data_test = splitobs(shuffleobs(data_train), at = 1.0)
    println("Training data size: ", length(data_train))
    data_train, mean_train, std_train = normalize_features(data_train)
    data_test, mean_test, mean_train = normalize_features(data_test, mean_train, std_train)
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
    jldsave(model_name; model_state)
    return model, mean_train, std_train
end
