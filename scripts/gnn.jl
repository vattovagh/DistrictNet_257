# helpers for train.jl

function decayed_learning_rate(epoch, init_lr)
    min_lr=0.0001
    decay=10
    decay_rate=0.9
    lr = max(init_lr * (decay_rate ^ floor(epoch / decay)), min_lr)
    return lr
end

function randomize_data(data, sample_monte_carlo)
    temp = []
    for item in data
        y = randomized_constructor(item[1].solution, sample_monte_carlo)
        push!(temp, (item[1], y))
    end
    return temp
end

function build_gnn_model(input_size, hidden_size)
    return Chain(
        GNNChain(
            GraphConv(input_size => hidden_size, relu; aggr = mean),
            GraphConv(hidden_size => hidden_size, relu; aggr = mean),
            GraphConv(hidden_size => hidden_size; aggr = mean),
            Dense(hidden_size => hidden_size, relu),
            Dense(hidden_size => hidden_size),
            Dense(hidden_size => 1),
            vec,
        ),
        x -> x.ndata.x,
    )
end

function train_GNN(model, data_train, data_test, hyperparams)

    # hyperparameter processing
    samples, samples_monte_carlo, batchsize, hidden_size, epochs, epsilon, lr = hyperparams

    # settings
    optim = Flux.Optimise.Adam(lr)
    LinearAlgebra.BLAS.set_num_threads(16)

    regularized_predictor = PerturbedAdditive(cmst_exact_solver; ε=epsilon, nb_samples=samples, seed=1234, is_parallel=true)
    loss_func = FenchelYoungLoss(regularized_predictor)

    flux_loss(x, y) = loss_func(model(x.gnn_graph), y; instance=x.instance, subgraphs=x.subgraphs)
    flux_loss_batch(batch) = sum(flux_loss(item[1], item[2]) for item in batch) 

    # training data processing
    data_train = randomize_data(data_train, samples_monte_carlo)
    train_dataset = Flux.DataLoader(data_train, batchsize=batchsize, shuffle=true)

    # training loop
    train_loss_prev = 0
    start_t = time()
    for epoch in 1:epochs
        Flux.trainmode!(model)
        optim.eta = decayed_learning_rate(epoch, lr)
        train_loss = 0
        for batch in train_dataset
            batch_loss = 0
            grads = Flux.gradient(Flux.params(model)) do
                batch_loss  = flux_loss_batch(batch)
            end
            train_loss += batch_loss 
            Flux.update!(optim, Flux.params(model), grads)
        end

        # Normalize training loss
        train_loss /= length(data_train)
        loss_change = train_loss_prev - train_loss
        println("Epoch $epoch: Loss = $train_loss, change = $loss_change")
        train_loss_prev = train_loss
    end
    end_t = time()

    @info "Training done. It took $(end_t - start_t) seconds."
    return model
end
