# Basic kernel that computes a partial Lehmer Matrix given dimensions N and k.
# It is "partial" in the sense that it does compute correct elements for a Lehmer Matrix,
# but it only does so for the first floor(k / i) elements of each row, 
# where i is the index of the given row.
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

    partialLehmerMatrix::Matrix{Float32} = zeros(Float32, (N, k))
    Threads.@threads for i = 1:N
        innerIndex::Int32 = 1
        for j = 1:k / i
            partialLehmerMatrix[i, innerIndex] = min(i, j) / max(i, j)
            innerIndex += 1
        end
    end

    display(partialLehmerMatrix)
end

main(ARGS)