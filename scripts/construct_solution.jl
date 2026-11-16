include("../src/construct.jl")
include("train.jl")

# now that we have training and testing (evaluating), we can construct results into a particular districting solution.

function construct_solution()
    train = train_and_fit()
    solution = find_solution_city(train...)
    path = "output/solution/Experiment/"
    if !isdir(path)
        mkpath(path)
    end
    filename = "output/solution/Experiment/output.txt"
    println("Solution found. Exporting...")
    export_solution(solution, filename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    construct_solution()
end
