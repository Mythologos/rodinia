using Printf

REC_LENGTH = 49   # size of a record in db
REC_WINDOW = 10   # number of records to read at a time
LATITUDE_POS = 28 # location of latitude coordinates in input record
OPEN = 10000      # initial value of nearest neighbors

const OUTPUT = haskey(ENV, "OUTPUT")

mutable struct Neighbor
    entry::String
    dist::Float64
end

# This program finds the k-nearest neighbors.
# Usage:	./nn <filelist> <num> <target latitude> <target longitude>
#			filelist: File with the filenames to the records
#			num: Number of nearest neighbors to find
#			target lat: Latitude coordinate for distance calculations
#			target long: Longitude coordinate for distance calculations
# The filelist and data are generated by hurricane_gen.c
# REC_WINDOW has been arbitrarily assigned; A larger value would allow more work
# for the threads.
function main(args)
    if length(args) < 4
        @printf(STDERR, "Invalid set of arguments\n");
        exit(-1)
    end

    flist = open(args[1], "r")
    #if (!flist) {
    #    println("error opening flist")
    #    exit(1);
    #}

    k = parse(Int, args[2])
    target_lat = parse(Float64, args[3])
    target_long = parse(Float64, args[4])

    neighbors = Array{Neighbor}(undef, k)
    #if (neighbors == NULL) {
    #    fprintf(stderr, "no room for neighbors\n");
    #    exit(0);
    #}

    # Initialize list of nearest neighbors to very large dist.
    for j in eachindex(neighbors)
        neighbors[j] = Neighbor("NULL", OPEN)
    end

    # Main processing
    dbname = chomp(readline(flist))
    #if (fscanf(flist, "%s\n", dbname) != 1) {
    #   fprintf(stderr, "error reading filelist\n");
    #   exit(0);
    #}

    fp = open(dbname, "r")
    #if (!fp) {
    #   printf("error opening flist\n");
    #   exit(1);
    #}

    done::Bool = false
    z = Array{Float32}(undef, REC_WINDOW)
    sandbox = Array{String}(undef, REC_WINDOW)

    while !done
        # Read in REC_WINDOW number of records
        rec_count = 0
        while !eof(fp) && (rec_count < REC_WINDOW)
            line = chomp(readline(fp))
            rec_count = rec_count + 1
            sandbox[rec_count] = line
        end

        if rec_count != REC_WINDOW
            close(fp)
            if eof(flist)
                done = 1
            else
                dbname = chomp(readline(flist))
                fp = open(dbname, "r")
            end
        end

        for i = 1:rec_count
            tmp_lat, tmp_long = map(x -> parse(Float32, x), split(sandbox[i][LATITUDE_POS:end]))
            z[i] = sqrt(((tmp_lat - target_lat) * (tmp_lat - target_lat)) +
                ((tmp_long - target_long) * (tmp_long - target_long)))
        end

        for i = 1:rec_count
            max_dist = -1
            max_idx = 1

            # Find a neighbor with greatest distance and take its spot.
            for j = 1:k
                if neighbors[j].dist > max_dist
                    max_dist = neighbors[j].dist
                    max_idx = j
                end
            end

            # Compare each record with max value to find the nearest neighbor.
            if z[i] < neighbors[max_idx].dist
                neighbors[max_idx].entry = sandbox[i]
                neighbors[max_idx].dist = z[i];
            end
        end
    end

    if OUTPUT
        out = open("output.txt", "w")
        @printf(out, "The %d nearest neighbors are:\n", k);
        for j in length(neighbors):-1:1
            if neighbors[j].dist != OPEN
                @printf(out, "%s --> %f\n", neighbors[j].entry, neighbors[j].dist)
            end
        end
        close(out)
    end

    print("The ", k, " nearest neighbors are:\n");
    for j in length(neighbors):-1:1
        if neighbors[j].dist != OPEN
            println(neighbors[j].entry, "; ", neighbors[j].dist)
        end
    end

    close(flist)
end

main(ARGS)
