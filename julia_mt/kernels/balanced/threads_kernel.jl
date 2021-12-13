# Basic kernel that computes a Lehmer Matrix given dimensions N and k.
# https://en.wikipedia.org/wiki/Lehmer_matrix
# Utilizes multithreading via the work-stealing of @threads (as of Julia 1.6).

function main(args)
    if length(args) != 2
        throw(ArgumentError(
            "An invalid number of arguments was given. 
            The correct number is two."
        ))
    end

    N::Int32 = parse(Int, args[1])
    k::Int32 = parse(Int, args[2])

    lehmerMatrix::Matrix{Float32} = zeros(Float32, (N, k))
    Threads.@threads for i = 1:N
        innerIndex::Int32 = 1
        for j = 1:k
            lehmerMatrix[i, j] = min(i, j) / max(i, j)
            innerIndex += 1
        end
    end

    display(lehmerMatrix)
end

main(ARGS)