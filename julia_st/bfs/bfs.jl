#!/usr/bin/env julia
using DelimitedFiles

const OUTPUT = haskey(ENV, "OUTPUT")

struct Node
    starting::Int32
    no_of_edges::Int32
end

function Usage()
    prog = basename(Base.source_path())
    println(stderr,"Usage: ", prog, " <input_file>")
    exit(1)
end

function parseline(::Type{T}, f) where {T<:Number}
    while true
        line = chomp(readline(f))
        if length(line) > 0
            return map(s -> parse(T, s), split(line))::Vector{T}
        end
    end
    return T[]
end

function main(args)
    if length(args) != 1
        Usage()
    end

    @info "Reading file"
    f = open(args[1],"r")

    # Read graph data
    no_of_nodes, = parseline(Int,f)
    h_graph_nodes = Array{Node, 1}(undef, no_of_nodes)
    for i in 1:no_of_nodes
        start, edgeno = parseline(Int, f)
        # add one to every node index to accomodate for 1-based arrays
        h_graph_nodes[i] = Node(start+1, edgeno)
    end

    # initialize node structures
    h_graph_mask = zeros(Bool,no_of_nodes)
    h_updating_graph_mask = zeros(Bool,no_of_nodes)
    h_graph_visited = zeros(Bool,no_of_nodes)

    # read the source node from the file (+1 to accomodate 1-based arrays)
    source, = parseline(Int,f)
    source += 1

    # set the source node as true in the mask
    h_graph_mask[source]=true
    h_graph_visited[source]=true

    # skip empty line
    edge_list_size, = parseline(Int,f)
    h_graph_edges_with_cost::Array{Int32,2} = readdlm(f,' ',UInt32; use_mmap=true, dims=(edge_list_size,2), quotes=false, comments=false)
    # add one to every node index to accomodate for 1-based arrays
    h_graph_edges = [h_graph_edges_with_cost[row,1]+1 for row in 1:size(h_graph_edges_with_cost,1)]

    close(f)

    h_cost = fill(Int32(-1),no_of_nodes)
    h_cost[source] = 0

    @info "Start traversing the tree"

    again = true
    while again
        # if no one changes this value then the loop stops
        again = false

        for maskiter in eachindex(h_graph_mask)
            if h_graph_mask[maskiter]
                h_graph_mask[maskiter] = false
                for nodeiter in h_graph_nodes[maskiter].starting:h_graph_nodes[maskiter].starting+h_graph_nodes[maskiter].no_of_edges-1
                    id = h_graph_edges[nodeiter]
                    if !h_graph_visited[id]
                        h_cost[id] = h_cost[maskiter]+1
                        h_updating_graph_mask[id]=true
                    end
                end
            end
        end

        for updatingmaskiter in eachindex(h_updating_graph_mask)
            if h_updating_graph_mask[updatingmaskiter]
                h_graph_mask[updatingmaskiter]=true
                h_graph_visited[updatingmaskiter]=true
                again=true
                h_updating_graph_mask[updatingmaskiter]=false
            end
        end
    end


    if OUTPUT
        f = open("output.txt","w")
        for i in eachindex(h_cost)
            println(f,i-1,") cost:",h_cost[i])
        end
        close(f)
        @info "Result stored in output.txt"
    end
end

main(ARGS)
