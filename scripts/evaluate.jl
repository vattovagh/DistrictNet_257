# Usage (from repo root): julia --project scripts/evaluate.jl [model_path]
# Scores a trained model on held-out cities and compares it with an untrained model.

using CSV, DataFrames

include("../dependencies/julia/instance.jl")
include("gnn.jl")

model_path = length(ARGS) >= 1 ? ARGS[1] : "models/DistrictNet257_$(NB_TRAIN).jld2"
saved = JLD2.load(model_path)

test_cities = aggregate_city_training_data(NB_TEST, NB_TRAIN + 1)
test_cities, _, _ = normalize_features(test_cities, saved["mean_train"], saved["stdev_train"])

input_size = size(saved["mean_train"], 1)
trained = build_gnn_model(input_size, HIDDEN_SIZE)
Flux.loadmodel!(trained, saved["model_state"])
untrained = build_gnn_model(input_size, HIDDEN_SIZE)

results = DataFrame(city = String[], optimal_cost = Float64[], trained_gap = Float64[], untrained_gap = Float64[])
for (item, _) in test_cities
    from_model = evaluate_city(trained, item)
    from_random = evaluate_city(untrained, item)
    push!(results, (item.instance.city_name, from_model.optimal, from_model.gap, from_random.gap))
end

mkpath("results")
CSV.write("results/evaluation.csv", results)
println(results)
println("Mean gap, trained: ", round(mean(results.trained_gap), digits = 2), "%   untrained: ", round(mean(results.untrained_gap), digits = 2), "%")
