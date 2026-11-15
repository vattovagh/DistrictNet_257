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
