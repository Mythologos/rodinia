#!/usr/bin/env julia
import Base.Threads

using Printf

const OUTPUT = haskey(ENV, "OUTPUT")
const MAX_LOOPS = 500


function main(args)
    if length(args) < 3
        println("Invalid set of arguments.\n")
        exit(-1)
    end
    
    numObjects::Int32, numAttributes::Int32 = 0, 0

    dataFile::IOStream = open(args[1], "r")
    # We open the data file and iterate over it to get the number of lines in it.
    numObjects = countlines(dataFile)

    # We reset the data file back to the beginning, read an item, and tabulate its number of elements.
    seekstart(dataFile)
    firstDataObject::String = readline(dataFile)
    numAttributes = size(split(firstDataObject), 1)

    # We define the main structure for the data for the rest of the computations.
    data::Vector{Vector{Float32}} = Vector{Vector{Float32}}(undef, numObjects)
        
    # We reset the data file back to the beginning to iterate over it and collect the data.
    seekstart(dataFile)
    for (line_index, line) in enumerate(eachline(dataFile))
        newVector::Vector{Float32} = Vector{Float32}(undef, numAttributes)
        newLine::Vector{String} = [string(token) for token in split(line)]
        for (token_index, token) in enumerate(newLine)
            newVector[token_index] = parse(Float32, token)
        end
        data[line_index] = newVector
    end

    # Since we're done with the file and have collected the full data, we close the file.
    close(dataFile)

    numClusters::Int32 = parse(Int32, args[2])
    thresholdValue::Int32 = parse(Int32, args[3])

    centroids::Vector{Vector{Float32}} = first(data, numClusters)
    membership::Vector{Int32} = zeros(Int32, numObjects)

    loopCount::Int16 = 0
    delta::Int32 = typemax(Int32)
    while delta > thresholdValue && loopCount < MAX_LOOPS
        new_centroids::Vector{Vector{Float32}} = [zeros(numAttributes) for _ in 1:numClusters]
        new_centroid_lengths::Vector{Int32} = [zero(Int32) for _ in 1:numClusters]

        delta = 0
        for (point_index, point) in enumerate(data)
            # For each point, we determine the nearest center--which determines which cluster it joins.
            new_membership::Int32 = find_nearest_center(point, centroids)
            if membership[point_index] != new_membership
                delta += 1
            end
            membership[point_index] = new_membership

            # We handle intermediary steps to compute new centroids later.
            new_centroids[new_membership] += data[point_index]
            new_centroid_lengths[new_membership] += 1
        end

        # We calculate the new centroids. If a cluster received no centroids, 
        # we use the same centroid from the previous iteration.
        for cluster_index in 1:numClusters
            if new_centroid_lengths[cluster_index] > 0
                centroids[cluster_index] = new_centroids[cluster_index] / new_centroid_lengths[cluster_index]
            end
        end
    end

    if OUTPUT
        out = open("output.txt", "w")
        for cluster_index in 1:numClusters
            @printf(out, "%d:", cluster_index)
            for attribute_index in 1:numAttributes
                @printf(out, " %f", centroids[cluster_index][attribute_index])
            end
            @printf(out, "\n")
        end
        close(out)
end

function get_squared_euclidean_distance(first::Vector{Float32}, second::Vector{Float32})
    return sum(abs2, first - second)
end

function find_nearest_center(point::Vector{Float32}, centroids::Vector{Vector{Float32}})
    nearest_center_index::Int32 = 0
    closest_distance = nothing
    for (centroid_index, centroid) in enumerate(centroids)
        current_distance::Float32 = get_squared_euclidean_distance(point, centroid)
        if isnothing(closest_distance) || closest_distance > current_distance
            closest_distance = current_distance
            nearest_center_index = centroid_index
        end
    end
    return nearest_center_index
end

main(ARGS)
