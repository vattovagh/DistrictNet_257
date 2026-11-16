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
        Flux.trainmode!(model)
        opt.eta = decayed_learning_rate(epoch, initial_lr)
        # Training
        train_loss = 0
        for batch in train_dataset
            batch_loss = 0
            grads = Flux.gradient(Flux.params(model)) do
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
