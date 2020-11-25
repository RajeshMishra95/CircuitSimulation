using LinearAlgebra
using DelimitedFiles
using Statistics
using LightGraphs
# using GraphPlot
# using Compose
# using Gadfly
# import Cairo, Fontconfig
using SparseArrays
using CSV

function error_lattice(d::Int64, cycles::Int64, initial_lattice)
    """
    Creats the adjacency list of the volume lattice using the initial lattice.
    """
    @assert cycles > 2
    final_lattice = deepcopy(initial_lattice)
    inc = d*(d+1)
    for j = 1:cycles-2
        for i in initial_lattice
            append!(final_lattice, i[1]+j*inc, i[2]+j*inc)
        end
    end
    return final_lattice
end