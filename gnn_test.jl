# gnn_smoke_test.jl
# Minimal end-to-end check: deps load -> build graph -> train -> evaluate.
# Run with: julia --project=. gnn_test.jl

using GraphNeuralNetworks
using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using Random, Statistics
using Test
using Pkg

println("Julia $(VERSION)")
for pkg in ("GraphNeuralNetworks", "Flux")
    deps = Pkg.dependencies()
    for (_, info) in deps
        info.name == pkg && println("  $pkg v$(info.version)")
    end
end

Random.seed!(42)

# ------------------------------------------------------------------
# 1. Synthetic node-classification graph (stochastic block model-ish)
# ------------------------------------------------------------------
nclasses     = 3
n_per_class  = 50
nfeat        = 8
n            = nclasses * n_per_class
labels       = repeat(1:nclasses, inner = n_per_class)

# Weak per-class signal in features, so graph structure actually helps
class_means = 0.5f0 .* randn(Float32, nfeat, nclasses)
X = class_means[:, labels] .+ randn(Float32, nfeat, n)

src, dst = Int[], Int[]
for i in 1:n, j in (i + 1):n
    p = labels[i] == labels[j] ? 0.10 : 0.005
    if rand() < p
        push!(src, i, j)   # undirected -> add both directions
        push!(dst, j, i)
    end
end

g = GNNGraph(src, dst; ndata = (; x = X))
Y = onehotbatch(labels, 1:nclasses)

@testset "graph construction" begin
    @test g.num_nodes == n
    @test g.num_edges == length(src)
    @test size(g.x) == (nfeat, n)
end

# Train / val / test node split
perm      = randperm(n)
train_idx = perm[1:90]
val_idx   = perm[91:120]
test_idx  = perm[121:end]

# ------------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------------
model = GNNChain(
    GCNConv(nfeat => 16, relu),
    Dropout(0.5),
    GCNConv(16 => nclasses),
)

accuracy(ŷ, idx) = mean(onecold(ŷ[:, idx]) .== labels[idx])

@testset "forward pass" begin
    ŷ = model(g, g.x)
    @test size(ŷ) == (nclasses, n)
    @test all(isfinite, ŷ)
end

# ------------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------------
opt_state = Flux.setup(Adam(0.01), model)

function train_loss(m)
    ŷ = m(g, g.x)
    return logitcrossentropy(ŷ[:, train_idx], Y[:, train_idx])
end

Flux.trainmode!(model)
initial_loss = train_loss(model)

for epoch in 1:200
    loss, grads = Flux.withgradient(train_loss, model)
    Flux.update!(opt_state, model, grads[1])
    if epoch == 1 || epoch % 50 == 0
        Flux.testmode!(model)
        ŷ = model(g, g.x)
        println("epoch $epoch  loss=$(round(loss, digits = 4))  ",
                "train_acc=$(round(accuracy(ŷ, train_idx), digits = 3))  ",
                "val_acc=$(round(accuracy(ŷ, val_idx), digits = 3))")
        Flux.trainmode!(model)
    end
end

# ------------------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------------------
Flux.testmode!(model)
ŷ = model(g, g.x)
final_loss = logitcrossentropy(ŷ[:, train_idx], Y[:, train_idx])
train_acc  = accuracy(ŷ, train_idx)
val_acc    = accuracy(ŷ, val_idx)
test_acc   = accuracy(ŷ, test_idx)

println("\nFinal: train=$(round(train_acc, digits=3))  ",
        "val=$(round(val_acc, digits=3))  test=$(round(test_acc, digits=3))")

@testset "training works" begin
    @test final_loss < initial_loss
    @test train_acc > 1 / nclasses + 0.2   # clearly better than chance
    @test test_acc  > 1 / nclasses         # at least beats chance on held-out nodes
end

println("\nAll checks passed.")
