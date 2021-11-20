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
    for (lineIndex, line) in enumerate(eachline(dataFile))
        newVector::Vector{Float32} = Vector{Float32}(undef, numAttributes)
        newLine::Vector{String} = [string(token) for token in split(line)]
        for (tokenIndex, token) in enumerate(newLine)
            newVector[tokenIndex] = parse(Float32, token)
        end
        data[lineIndex] = newVector
    end

    # Since we're done with the file and have collected the full data, we close the file.
    close(dataFile)

    numClusters::Int32 = parse(Int32, args[2])
    thresholdValue::Int32 = parse(Int32, args[3])

    centroids::Vector{Vector{Float32}} = first(data, numClusters)
    membership::Vector{Int32} = zeros(Int32, numObjects)

    # These data structures are individualized for each thread.
    # They are the only ones that should actively be assigned to during multithreading,
    # unless the variables are temporary.
    threadedCentroids::Vector{Vector{Vector{Float32}}} = [[zeros(Float32, numAttributes) for _ in 1:numClusters] for _ in 1:Threads.nthreads()]
    threadedCentroidLengths::Vector{Vector{Int32}} = [[zero(Int32) for _ in 1:numClusters] for _ in 1:Threads.nthreads()]
    threadedDelta::Vector{Int32} = zeros(Int32, Threads.nthreads())

    newCentroids::Vector{Vector{Float32}} = [zeros(numAttributes) for _ in 1:numClusters]
    newCentroidLengths::Vector{Int32} = [zero(Int32) for _ in 1:numClusters]

    loopCount::Int16 = 0
    delta::Int32 = typemax(Int32)
    while delta > thresholdValue && loopCount < MAX_LOOPS
        delta = 0
        Threads.@threads for (pointIndex, point) in collect(enumerate(data))
            # For each point, we determine the nearest center--which determines which cluster it joins.
            newMembership::Int32 = find_nearest_center(point, centroids)
            if membership[pointIndex] != newMembership
                threadedDelta[Threads.threadid()] += 1
            end
            membership[pointIndex] = newMembership

            # We handle intermediary steps to compute new centroids later.
            threadedCentroids[Threads.threadid()][newMembership] += data[pointIndex]
            threadedCentroidLengths[Threads.threadid()][newMembership] += 1
        end

        # We calculate the new centroids. If a cluster received no centroids, 
        # we use the same centroid from the previous iteration.
        for clusterIndex in 1:numClusters
            for threadIndex in 1:Threads.nthreads()
                newCentroidLengths[clusterIndex] += threadedCentroidLengths[threadIndex][clusterIndex]
                threadedCentroidLengths[threadIndex][clusterIndex] = 0

                newCentroids[clusterIndex] += threadedCentroids[threadIndex][clusterIndex]
                threadedCentroids[threadIndex][clusterIndex] = zeros(Float32, numAttributes)
            end
        end

        for clusterIndex in 1:numClusters
            if newCentroidLengths[clusterIndex] > 0
                centroids[clusterIndex] = newCentroids[clusterIndex] / newCentroidLengths[clusterIndex]
            end
            newCentroids[clusterIndex] = zeros(Float32, numAttributes)
            newCentroidLengths[clusterIndex] = 0
        end
        delta = sum(threadedDelta)
        threadedDelta = zeros(Threads.nthreads())
        loopCount += 1
    end

    if OUTPUT
        out = open("output.txt", "w")
        for clusterIndex in 1:numClusters
            @printf(out, "%d:", clusterIndex)
            for attributeIndex in 1:numAttributes
                @printf(out, " %f", centroids[clusterIndex][attributeIndex])
            end
            @printf(out, "\n")
        end
        close(out)
    end
end

function get_squared_euclidean_distance(first::Vector{Float32}, second::Vector{Float32})
    return sum(abs2, first - second)
end

function find_nearest_center(point::Vector{Float32}, centroids::Vector{Vector{Float32}})
    nearestCenterIndex::Int32 = 0
    closestDistance = nothing
    for (centroidIndex, centroid) in enumerate(centroids)
        currentDistance::Float32 = get_squared_euclidean_distance(point, centroid)
        if isnothing(closestDistance) || closestDistance > currentDistance
            closestDistance = currentDistance
            nearestCenterIndex = centroidIndex
        end
    end
    return nearestCenterIndex
end

main(ARGS)
