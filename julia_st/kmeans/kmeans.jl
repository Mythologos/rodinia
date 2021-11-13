# Based on the OpenMP version of the k-means algorithm, this algorithm should do the following:
# It should have a command line. This command line accepts arguments of:
# (1) the filename for an input file containing data to examine.
# (2) an indication of whether the file is a binary file or not. [Omitted for now, as there doesn't seem to be an explicit need to use this file type.]
# (3) the number of clusters (with a default value of 5; the k in k-means)
# (4) the threshold value for clustering
# (5) the number of threads to use [Omitted, since this is the single-threaded version, and we only need one.]
# This k-means algorithm is regular. Although mentions of a fuzzy c-means algorithm are in the code,
# the code ultimately represents a traditional k-means algorithm.
#
# To perform this algorithm, there are a few steps:
# (1) From a set of initial centroids--central points--of size k, we calculate the distance between a point x
# and each centroid. The calculations here can be independent, so this is a place where parallelization can occur.
# (2) For each point, we determine what cluster that point is a part of by determining the closest centroid.
# Points are grouped by these centroids. When done grouping, we calculate the overall "error" of the clustering process.
# If the error is less than the threshold value, we're done and can return the clusters. Otherwise,
# we aren't done and we calculate new centroids based on the clusters (e.g., the mean of the points in each cluster),
# and we repeat the process with the new centroids but the same points. 
# The algorithm continues until the error is under the threshold value.

# The actual C code proceeds by first allocating space for all data and loading it in.
# After that, it performs the actual algorithm using cluster.c and kmeans_clustering.c.
# They use Euclidean distance as their distance metric 
# (with a squared version, since the order is more important than the magnitude.
# Then, they utilize the first k data points to decide initial clusters.
# When performing k-means, it seems like they proceed to calculate the threshold by recording cluster membership.
# In other words, the first round will proceed to the second one no matter what. When the second round comes,
# the first membership list and the second membership list are compared point by point.
# If the number of changes is less than the threshold, the iteration ends. Otherwise, it doesn't,
# and the new memberships are used to tabulate change for the next round until convergence occurs.
# The OpenMP code enjoys performing running tallies of data and manipulating each point in every way possible
# so as to parallelize various actions to the greatest extent. For example, it computes partial sums
# for clusters in preparation for tabulating new centroids out of the averages of the contained points.

# Note that, to confirm convergence, they set a limit for the number of loops to 500.

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
    data::Array{Float32, 2} = zeros(Float32, (numObjects, numAttributes))
        
    # We reset the data file back to the beginning to iterate over it and collect the data.
    seekstart(dataFile)
    for (line_index, line) in enumerate(eachline(dataFile))
        newLine::Array{String, 1} = [string(token) for token in split(line)]
        for (token_index, token) in enumerate(newLine)
            data[line_index, token_index] = parse(Float32, token)
        end
    end

    # Since we're done with the file and have collected the full data, we close the file.
    close(dataFile)

    numClusters::Int32 = parse(Int32, args[2])
    thresholdValue::Int32 = parse(Int32, args[3])
end

main(ARGS)
