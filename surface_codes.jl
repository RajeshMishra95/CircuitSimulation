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
using DataFrames

mutable struct QuantumState
    tableau::Matrix
    qubits::Int64

    QuantumState(qubits::Int64) = new(hcat(Matrix{Int64}(I,2*qubits,2*qubits),zeros(Int64,2*qubits,1)),qubits)
end 

function apply_h!(QS::QuantumState, qubit::Int64)
    for i = 1:2*QS.qubits
        QS.tableau[i,2*QS.qubits+1] = xor(QS.tableau[i,2*QS.qubits+1],QS.tableau[i,qubit]*QS.tableau[i,qubit+QS.qubits])
        QS.tableau[i,qubit], QS.tableau[i,qubit+QS.qubits] = QS.tableau[i,qubit+QS.qubits], QS.tableau[i,qubit]
    end
end

function apply_s!(QS::QuantumState, qubit::Int64)
    for i = 1:2*QS.qubits
        QS.tableau[i,2*QS.qubits+1] = xor(QS.tableau[i,2*QS.qubits+1],QS.tableau[i,qubit]*QS.tableau[i,qubit+QS.qubits])
        QS.tableau[i,qubit+QS.qubits] = xor(QS.tableau[i,qubit+QS.qubits], QS.tableau[i,qubit])
    end
end

function apply_cnot!(QS::QuantumState, control_qubit::Int64, target_qubit::Int64)
    for i = 1:2*QS.qubits
        QS.tableau[i,2*QS.qubits+1] = xor(QS.tableau[i,2*QS.qubits+1],QS.tableau[i,control_qubit]*QS.tableau[i,target_qubit+QS.qubits]*xor(QS.tableau[i,target_qubit],xor(QS.tableau[i,control_qubit+QS.qubits],1)))
        QS.tableau[i,target_qubit] = xor(QS.tableau[i,target_qubit],QS.tableau[i,control_qubit])
        QS.tableau[i,control_qubit+QS.qubits] = xor(QS.tableau[i,control_qubit+QS.qubits],QS.tableau[i,target_qubit+QS.qubits])
    end
end

function apply_z!(QS::QuantumState, qubit::Int64)
    apply_s!(QS, qubit)
    apply_s!(QS, qubit)
end

function apply_x!(QS::QuantumState, qubit::Int)
    apply_h!(QS, qubit)
    apply_z!(QS, qubit)
    apply_h!(QS, qubit)
end

function apply_y!(QS::QuantumState, qubit::Int64)
    apply_z!(QS, qubit)
    apply_x!(QS, qubit)
end

function apply_Rz!(QS::QuantumState, qubit::Int64)
    measurement = measure!(QS, qubit)
    if measurement != [1,0]
        apply_x!(QS, qubit)
    end
end

function func_g(x1::Int64,z1::Int64,x2::Int64,z2::Int64)
    if (x1 == 0) && (z1 == 0)
        return 0
    elseif (x1 == 1) && (z1 == 1)
        return z2 - x2
    elseif (x1 == 1) && (z1 == 0)
        return z2*(2*x2 - 1)
    else 
        return x2*(1 - 2*z2)
    end
end

function rowsum!(QS::QuantumState, h::Int64, i::Int64)
    value = 2*QS.tableau[h,2*QS.qubits+1] + 2*QS.tableau[i,2*QS.qubits+1]
    for j = 1:QS.qubits
        value += func_g(QS.tableau[i,j], QS.tableau[i,j+QS.qubits], QS.tableau[h,j], QS.tableau[h,j+QS.qubits])
    end

    if value%4 == 0
        QS.tableau[h,2*QS.qubits+1] = 0
    elseif value%4 == 2
        QS.tableau[h,2*QS.qubits+1] = 1
    end
    for j = 1:QS.qubits
        QS.tableau[h,j] = xor(QS.tableau[i,j],QS.tableau[h,j])
        QS.tableau[h,j+QS.qubits] = xor(QS.tableau[i,j+QS.qubits],QS.tableau[h,j+QS.qubits])
    end
end

function prob_measurement!(QS::QuantumState, qubit::Int64)
    is_random = false
    p::Int64 = 0
    for i = QS.qubits + 1:2*QS.qubits
        if QS.tableau[i,qubit] == 1
            is_random = true
            p = i
            break
        end
    end

    if is_random
        return [0.5,0.5,p]
    else
        QS.tableau = vcat(QS.tableau, zeros(Int64,1,2*QS.qubits+1))
        for i = 1:QS.qubits
            if QS.tableau[i,qubit] == 1
                rowsum!(QS,2*QS.qubits+1,i+QS.qubits)
            end
        end
        if QS.tableau[2*QS.qubits+1,2*QS.qubits+1] == 0
            QS.tableau = QS.tableau[1:2*QS.qubits,:]
            return [1,0]
        else
            QS.tableau = QS.tableau[1:2*QS.qubits,:]
            return [0,1]
        end
    end
end

function measure!(QS::QuantumState, qubit::Int64)
    prob_measure = prob_measurement!(QS, qubit)
    if prob_measure == [1,0]
        return +1
    elseif prob_measure == [0,1]
        return -1
    else
        for i = 1:2*QS.qubits
            if QS.tableau[i,qubit] == 1 & (i != Int64(prob_measure[3]))
                rowsum!(QS,i,Int64(prob_measure[3]))
            end
        end
        QS.tableau[Int64(prob_measure[3])-QS.qubits,:] = QS.tableau[Int64(prob_measure[3]),:]
        QS.tableau[Int64(prob_measure[3]),:] = zeros(Int64,1,2*QS.qubits+1)
        QS.tableau[Int64(prob_measure[3]),2*QS.qubits+1] = rand((0,1))
        QS.tableau[Int64(prob_measure[3]),QS.qubits+qubit] = 1
        if QS.tableau[Int64(prob_measure[3]),2*QS.qubits+1] == 0
            return +1
        elseif QS.tableau[Int64(prob_measure[3]),2*QS.qubits+1] == 1
            return -1
        end
    end
end

function oneNorm(vector::Array{Float64,1})
    one_norm_vector = 0.0
    for i in vector
        one_norm_vector += abs(i)
    end
    return one_norm_vector
end

function probDistribution(quasi_prob::Array{Float64,1})
    prob = zeros(0)
    quasi_prob_norm = oneNorm(quasi_prob)
    for i in quasi_prob
        append!(prob, abs(i)/quasi_prob_norm)
    end
    return prob
end

function samplingDistribution(pdf::Array{Float64,1})
    cdf = zeros(1)
    for i = 1:length(pdf)
        append!(cdf,cdf[i]+pdf[i])
    end

    random_value = rand()
    idx = 0

    for i = 1:length(cdf)
        if random_value <= cdf[i]
            idx = i - 1
            break
        end
    end
    return idx
end

function generate_connections(n::Int64)
    d::Int64 = n
    dim::Int64 = 2*d - 1
    total_qubits::Int64 = dim*dim
    connections_dict = Dict{Int64,Array{Int64,1}}()
    for i = 1:total_qubits
        list_connections::Array{Int64,1} = []
        if i%dim == 0
            if i-dim > 0
                append!(list_connections,i-dim)
            end
            append!(list_connections,i-1)
            if i+dim < total_qubits + 1
                append!(list_connections,i+dim)
            end
        elseif i%dim == 1
            if i-dim > 0
                append!(list_connections,i-dim)
            end
            append!(list_connections,i+1)
            if i+dim < total_qubits+1
                append!(list_connections,i+dim)
            end
        else
            if i-dim > 0
                append!(list_connections,i-dim)
            end
            append!(list_connections,i-1)
            append!(list_connections,i+1)
            if i+dim < total_qubits+1
                append!(list_connections,i+dim)
            end
        end
        push!(connections_dict, i => list_connections)
    end
    return connections_dict
end

function generate_data_qubits(total_qubits::Int64)
    data_qubits::Array{Int64,1} = []
    for i = 1:total_qubits
        if i%2 != 0
            append!(data_qubits, i)
        end
    end
    return data_qubits
end

function generate_x_ancillas(d::Int64, total_qubits::Int64)
    x_ancillas::Array{Int64,1} = []
    for i = 1:d-1
        append!(x_ancillas, 2*i)
    end
    for j in x_ancillas
        if j + 4*d - 2 < total_qubits
            append!(x_ancillas, j + 4*d - 2)
        end
    end
    return x_ancillas
end

function generate_z_ancillas(d::Int64, total_qubits::Int64)
    z_ancillas::Array{Int64,1} = [2*d]
    for i = 1:d-1
        append!(z_ancillas, z_ancillas[end] + 2)
    end
    for j in z_ancillas
        if j + 4*d - 2  < total_qubits
            append!(z_ancillas, j + 4*d - 2)
        end
    end
    return z_ancillas
end

function gateOperation!(QS::QuantumState, gate::Int64, qubit::Int64)
    if gate == 2
        apply_z!(QS, qubit)
    elseif gate == 3
        apply_Rz!(QS, qubit)
    end
end

function prep_x_ancillas!(QS::QuantumState, x_ancilla_list::Vector{Int64})
    for i in x_ancilla_list
        apply_h!(QS,i)
    end
end

function north_z_ancillas!(QS::QuantumState, graph::Dict, z_ancilla_list::Vector{Int64})
    for j in z_ancilla_list
        if j-1 in graph[j]
            apply_cnot!(QS, j-1, j)
        end
    end
end

function north_x_ancillas!(QS::QuantumState, graph::Dict, x_ancilla_list::Vector{Int64})
    for j in x_ancilla_list
        if j-1 in graph[j]
            apply_cnot!(QS, j, j-1)
        end
    end
end

function west_z_ancillas!(QS::QuantumState, d::Int64, graph::Dict, z_ancilla_list::Vector{Int64})
    for j in z_ancilla_list
        if j-(2*d-1) in graph[j]
            apply_cnot!(QS, j-(2*d-1), j)
        end
    end
end

function west_x_ancillas!(QS::QuantumState, d::Int64, graph::Dict, x_ancilla_list::Vector{Int64})
    for j in x_ancilla_list
        if j-(2*d-1) in graph[j]
            apply_cnot!(QS, j, j-(2*d-1))
        end
    end
end

function east_z_ancillas!(QS::QuantumState, d::Int64, graph::Dict, z_ancilla_list::Vector{Int64})
    for j in z_ancilla_list
        if j+(2*d-1) in graph[j]
            apply_cnot!(QS, j+(2*d-1), j)
        end
    end
end

function east_x_ancillas!(QS::QuantumState, d::Int64, graph::Dict, x_ancilla_list::Vector{Int64})
    for j in x_ancilla_list
        if j+(2*d-1) in graph[j]
            apply_cnot!(QS, j, j+(2*d-1))
        end
    end
end

function south_z_ancillas!(QS::QuantumState, graph::Dict, z_ancilla_list::Vector{Int64})
    for j in z_ancilla_list
        if j+1 in graph[j]
            apply_cnot!(QS, j+1, j)
        end
    end
end

function south_x_ancillas!(QS::QuantumState, graph::Dict, x_ancilla_list::Vector{Int64})
    for j in x_ancilla_list
        if j+1 in graph[j]
            apply_cnot!(QS, j, j+1)
        end
    end
end

function apply_noise!(QS::QuantumState, noise::Int64, qubit::Int64)
    if noise == 2
        apply_x!(QS, qubit)
    elseif noise == 3 
        apply_y!(QS, qubit)
    elseif noise == 4
        apply_z!(QS, qubit)
    elseif noise == 5
        apply_s!(QS, qubit)
    elseif noise == 6
        apply_Rz!(QS, qubit)
    end
end

function measurement_circuit!(QS::QuantumState, d::Int64, graph::Dict, 
    x_ancilla_list::Vector{Int64}, z_ancilla_list::Vector{Int64})
    # prepare the x_ancillas in the |+> state
    prep_x_ancillas!(QS, x_ancilla_list)

    # carry out the measurement circuits
    # North
    north_z_ancillas!(QS, graph, z_ancilla_list)

    north_x_ancillas!(QS, graph, x_ancilla_list)

    # West
    west_z_ancillas!(QS, d, graph, z_ancilla_list)

    west_x_ancillas!(QS, d, graph, x_ancilla_list)

    # East
    east_z_ancillas!(QS, d, graph, z_ancilla_list)

    east_x_ancillas!(QS, d, graph, x_ancilla_list)

    # South
    south_z_ancillas!(QS, graph, z_ancilla_list)

    south_x_ancillas!(QS, graph, x_ancilla_list) 

    # measurement of the x_ancillas is the x basis
    for l in x_ancilla_list
        apply_h!(QS,l)
    end
end

function find_fault(d::Int64, timestep::Vector{Int64}, timestep0::Vector{Int64}, 
    timestep1::Vector{Int64}, timestep2::Vector{Int64}, 
    x_ancilla_list::Vector{Int64}, z_ancilla_list::Vector{Int64})
    x_ancilla_values::Vector{Int64} = []
    z_ancilla_values::Vector{Int64} = []
    x_fault::Vector{Int64} = []
    z_fault::Vector{Int64} = []

    # Errors appearing in first measurement cycle
    for i in x_ancilla_list
        append!(x_ancilla_values, timestep1[i]*timestep[i])
    end

    for i in z_ancilla_list
        append!(z_ancilla_values, timestep1[i]*timestep[i])
    end

    for i = 1:2*d
        append!(x_ancilla_values, 0)
    end

    for i = 1:2*d
        append!(z_ancilla_values, 0)
    end

    # Errors appearing in second measurement cycle
    for i in x_ancilla_list
        append!(x_ancilla_values, timestep2[i]*timestep0[i])
    end
    
    for i in z_ancilla_list
        append!(z_ancilla_values, timestep2[i]*timestep0[i])
    end

    if -1 in x_ancilla_values
        x_fault = findall(x -> x == -1, x_ancilla_values)
    end
    if -1 in z_ancilla_values
        z_fault = findall(z -> z == -1, z_ancilla_values)
    end

    return (x_fault, z_fault)
end

function generate_fault_graph(QS::QuantumState, d::Int64, graph::Dict, 
    x_ancilla_list::Vector{Int64}, z_ancilla_list::Vector{Int64},
    initial_measurement_values::Vector{Int64})

    # initialize 

    which_ancilla::Vector{Int64} = [0,1]
    num_ancillas::UnitRange{Int64} = range(1, length = d*(d-1))
    x_depth::Vector{Int64} = [1,2,3,4,5,6]
    z_depth::Vector{Int64} = [2,3,4,5]
    error::Vector{Int64} = [1,2,3,4] # 1 -> I, 2 -> X, 3 -> Y, 4 -> Z 
    G_x = SimpleGraph(2*length(x_ancilla_list)+4*d)
    G_z = SimpleGraph(2*length(z_ancilla_list)+4*d)

    # noisy measurement circuit (enumerate through all possible single errors)
    for i in which_ancilla
        if i == 0
            # error in one of the x_ancilla circuits
            for j in num_ancillas
                # error during one of the 6 timesteps
                for k in x_depth
                    if k == 1
                        # noisy hadamard gate
                        for l in error
                            qs = deepcopy(QS)
                            measurement_cycle = deepcopy(initial_measurement_values)
                            measurement_cycle0 = deepcopy(initial_measurement_values)
                            measurement_cycle1 = deepcopy(initial_measurement_values)
                            measurement_cycle2 = deepcopy(initial_measurement_values)
                            # preparation measurement circuit
                            measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                            # ancilla measurement for preparation
                            for i in x_ancilla_list
                                measurement_cycle0[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle0[j] = measure!(qs,j)
                            end
                            prep_x_ancillas!(qs, x_ancilla_list)
                            apply_noise!(qs, l, x_ancilla_list[j])
                            # carry out the measurement circuits
                            north_z_ancillas!(qs, graph, z_ancilla_list)
                            north_x_ancillas!(qs, graph, x_ancilla_list)
                            west_z_ancillas!(qs, d, graph, z_ancilla_list)
                            west_x_ancillas!(qs, d, graph, x_ancilla_list)
                            east_z_ancillas!(qs, d, graph, z_ancilla_list)
                            east_x_ancillas!(qs, d, graph, x_ancilla_list)
                            south_z_ancillas!(qs, graph, z_ancilla_list)
                            south_x_ancillas!(qs, graph, x_ancilla_list) 
                            for m in x_ancilla_list
                                apply_h!(qs,m)
                            end
                            # ancilla measurement for cycle1
                            for i in x_ancilla_list
                                measurement_cycle1[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle1[j] = measure!(qs,j)
                            end

                            # ideal measurement circuit
                            measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                            # ancilla measurement for cycle2
                            for i in x_ancilla_list
                                measurement_cycle2[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle2[j] = measure!(qs,j)
                            end
                            edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                            measurement_cycle2, x_ancilla_list, z_ancilla_list)
                            if length(edges[1]) == 2
                                add_edge!(G_x, edges[1][1], edges[1][2])
                            end
                            if length(edges[2]) == 2
                                add_edge!(G_z, edges[2][1], edges[2][2])
                            end
                        end
                    elseif k == 2
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                if x_ancilla_list[j] - 1 in graph[x_ancilla_list[j]]
                                    apply_noise!(qs, l, x_ancilla_list[j])
                                    apply_noise!(qs, m, x_ancilla_list[j] - 1)
                                end
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 3 
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                if x_ancilla_list[j] - (2*d - 1) in graph[x_ancilla_list[j]]
                                    apply_noise!(qs, l, x_ancilla_list[j])
                                    apply_noise!(qs, m, x_ancilla_list[j] - (2*d - 1))
                                end
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 4
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                if x_ancilla_list[j] + (2*d - 1) in graph[x_ancilla_list[j]]
                                    apply_noise!(qs, l, x_ancilla_list[j])
                                    apply_noise!(qs, m, x_ancilla_list[j] + (2*d - 1))
                                end
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list) 
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 5
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list)
                                if x_ancilla_list[j] + 1 in graph[x_ancilla_list[j]]
                                    apply_noise!(qs, l, x_ancilla_list[j])
                                    apply_noise!(qs, m, x_ancilla_list[j] + 1)
                                end 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    else
                        # noisy hadamard gate
                        for l in error
                            qs = deepcopy(QS)
                            measurement_cycle = deepcopy(initial_measurement_values)
                            measurement_cycle0 = deepcopy(initial_measurement_values)
                            measurement_cycle1 = deepcopy(initial_measurement_values)
                            measurement_cycle2 = deepcopy(initial_measurement_values)
                            
                            # preparation measurement circuit
                            measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                            # ancilla measurement for preparation
                            for i in x_ancilla_list
                                measurement_cycle0[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle0[j] = measure!(qs,j)
                            end
                            prep_x_ancillas!(qs, x_ancilla_list)
                            # carry out the measurement circuits
                            north_z_ancillas!(qs, graph, z_ancilla_list)
                            north_x_ancillas!(qs, graph, x_ancilla_list)
                            west_z_ancillas!(qs, d, graph, z_ancilla_list)
                            west_x_ancillas!(qs, d, graph, x_ancilla_list)
                            east_z_ancillas!(qs, d, graph, z_ancilla_list)
                            east_x_ancillas!(qs, d, graph, x_ancilla_list)
                            south_z_ancillas!(qs, graph, z_ancilla_list)
                            south_x_ancillas!(qs, graph, x_ancilla_list) 
                            for m in x_ancilla_list
                                apply_h!(qs,m)
                            end
                            apply_noise!(qs, l, x_ancilla_list[j])
                            # ancilla measurement for cycle1
                            for i in x_ancilla_list
                                measurement_cycle1[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle1[j] = measure!(qs,j)
                            end

                            # ideal measurement circuit
                            measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                            # ancilla measurement for cycle2
                            for i in x_ancilla_list
                                measurement_cycle2[i] = measure!(qs,i)
                            end

                            for j in z_ancilla_list
                                measurement_cycle2[j] = measure!(qs,j)
                            end
                            edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                            measurement_cycle2, x_ancilla_list, z_ancilla_list)
                            if length(edges[1]) == 2
                                add_edge!(G_x, edges[1][1], edges[1][2])
                            end
                            if length(edges[2]) == 2
                                add_edge!(G_z, edges[2][1], edges[2][2])
                            end
                        end 
                    end
                end
            end
        else    
            # error in one of the z_ancilla circuits
            for j in num_ancillas
                # error during one of the 6 timesteps
                for k in z_depth
                    if k == 2
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                if z_ancilla_list[j] - 1 in graph[z_ancilla_list[j]]
                                    apply_noise!(qs, l, z_ancilla_list[j])
                                    apply_noise!(qs, m, z_ancilla_list[j] - 1)
                                end
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 3 
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                if z_ancilla_list[j] - (2*d - 1) in graph[z_ancilla_list[j]]
                                    apply_noise!(qs, l, z_ancilla_list[j])
                                    apply_noise!(qs, m, z_ancilla_list[j] - (2*d - 1))
                                end
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end

                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 4
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                if z_ancilla_list[j] + (2*d - 1) in graph[z_ancilla_list[j]]
                                    apply_noise!(qs, l, z_ancilla_list[j])
                                    apply_noise!(qs, m, z_ancilla_list[j] + (2*d - 1))
                                end
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list) 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end
                    elseif k == 5
                        for l in error
                            for m in error
                                qs = deepcopy(QS)
                                measurement_cycle = deepcopy(initial_measurement_values)
                                measurement_cycle0 = deepcopy(initial_measurement_values)
                                measurement_cycle1 = deepcopy(initial_measurement_values)
                                measurement_cycle2 = deepcopy(initial_measurement_values)
                                
                                # preparation measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for preparation
                                for i in x_ancilla_list
                                    measurement_cycle0[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle0[j] = measure!(qs,j)
                                end
                                prep_x_ancillas!(qs, x_ancilla_list)
                                # carry out the measurement circuits
                                north_z_ancillas!(qs, graph, z_ancilla_list)
                                north_x_ancillas!(qs, graph, x_ancilla_list)
                                west_z_ancillas!(qs, d, graph, z_ancilla_list)
                                west_x_ancillas!(qs, d, graph, x_ancilla_list)
                                east_z_ancillas!(qs, d, graph, z_ancilla_list)
                                east_x_ancillas!(qs, d, graph, x_ancilla_list)
                                south_z_ancillas!(qs, graph, z_ancilla_list)
                                south_x_ancillas!(qs, graph, x_ancilla_list)
                                if z_ancilla_list[j] + 1 in graph[z_ancilla_list[j]]
                                    apply_noise!(qs, l, z_ancilla_list[j])
                                    apply_noise!(qs, m, z_ancilla_list[j] + 1)
                                end 
                                for m in x_ancilla_list
                                    apply_h!(qs,m)
                                end
                                # ancilla measurement for cycle1
                                for i in x_ancilla_list
                                    measurement_cycle1[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle1[j] = measure!(qs,j)
                                end

                                # ideal measurement circuit
                                measurement_circuit!(qs, d, graph, x_ancilla_list, z_ancilla_list)

                                # ancilla measurement for cycle2
                                for i in x_ancilla_list
                                    measurement_cycle2[i] = measure!(qs,i)
                                end

                                for j in z_ancilla_list
                                    measurement_cycle2[j] = measure!(qs,j)
                                end
                                edges = find_fault(d, measurement_cycle, measurement_cycle0, measurement_cycle1, 
                                measurement_cycle2, x_ancilla_list, z_ancilla_list)
                                if length(edges[1]) == 2
                                    add_edge!(G_x, edges[1][1], edges[1][2])
                                end
                                if length(edges[2]) == 2
                                    add_edge!(G_z, edges[2][1], edges[2][2])
                                end
                            end
                        end 
                    end
                end
            end
        end
    end

    for i in range(1, length = d)
        add_edge!(G_x, i, d*(d-1) + i)
        add_edge!(G_z, i, d*(d-1) + i)
    end

    for i in range(1+d*(d-2), length = d)
        add_edge!(G_x, i, d*(d-1) + i)
        add_edge!(G_z, i, d*(d-1) + i)
    end

    for i in range(1+d*(d+1), length = d)
        add_edge!(G_x, i, d*(d-1) + i)
        add_edge!(G_z, i, d*(d-1) + i)
    end

    for i in range(1+d*(d-2)+d*(d+1), length = d)
        add_edge!(G_x, i, d*(d-1) + i)
        add_edge!(G_z, i, d*(d-1) + i)
    end

    CSC_G_x = findnz(adjacency_matrix(G_x))
    CSC_G_z = findnz(adjacency_matrix(G_z))
    open("CSC_G_vertex.txt", "w") do f
        for i = 1:length(CSC_G_x[1])
            println(f, (CSC_G_x[1][i], CSC_G_x[2][i]))
        end
    end

    open("CSC_G_plaquette.txt", "w") do f
        for i = 1:length(CSC_G_z[1])
            println(f, (CSC_G_z[1][i], CSC_G_z[2][i]))
        end
    end

    # display(adjacency_matrix(G_x))
    # display(adjacency_matrix(G_z))
end

function generate_fault_nodes(d::Int64, num_cycles::Int64, measurement_cycles::Array{Array{Int64,1},1},
    x_ancilla_list::Vector{Int64}, z_ancilla_list::Vector{Int64})
    """
    Generates a list of nodes (vertex and plaquettes) with faults 
    numbered in the form of the volume lattice. Measurement cycles
    are measurement values for multiple cycles. 
    measurement_cycle[1]: initial values corresponding to +1
    measurement_cycle[2]: values of 1st cycle
    measurement_cycle[3]: values of 2nd cycle
    and so on ...
    """
    vertex_fault_list::Vector{Int64} = []
    for i = 1:length(measurement_cycles)-1
        for j = 1:length(x_ancilla_list)
            if measurement_cycles[i][x_ancilla_list[j]]*measurement_cycles[i+1][x_ancilla_list[j]] == -1
                append!(vertex_fault_list, j+(i-1)*d*(d+1))
            end
        end
    end

    plaquette_fault_list::Vector{Int64} = []
    for i = 1:length(measurement_cycles)-1
        for j = 1:length(z_ancilla_list)
            if measurement_cycles[i][z_ancilla_list[j]]*measurement_cycles[i+1][z_ancilla_list[j]] == -1
                append!(plaquette_fault_list, j+(i-1)*d*(d+1))
            end
        end
    end

    return vertex_fault_list, plaquette_fault_list
end

# function error_lattice(d::Int64, cycles::Int64, initial_lattice::Array{Array{Int64},1})
#     """
#     Creats the adjacency list of the volume lattice using the initial lattice.
#     """
#     @assert cycles > 2
#     final_lattice = deepcopy(initial_lattice)
#     inc = d*(d+1)
#     for j = 1:cycles-2
#         for i in initial_lattice
#             append!(final_lattice, i[1]+j*inc, i[2]+j*inc)
#         end
#     end
#     return final_lattice
# end

# function build_lattice_graph(d::Int64, cycles::Int64, edges::Array{Array{Int64,1}})
#     """Generates a graph given the arguments."""
#     G = SimpleGraph(d*(d+1)*cycles)
#     for i in edges
#         add_edge!(G, i[1], i[2])
#     end
#     return G
# end

# function generate_shortest_path_graph(d::Int64, cycles::Int64, volume_lattice::Array{Array{Int64,1}},
#     fault_nodes::Vector{Int64})
#     """
#     Takes the fault node and generates a graph containing the fault nodes and the 
#     shortest distance between each of the fault nodes and also between the fault nodes
#     their corresponding spatial and temporal ghost nodes. Here, ghost can be shared by 
#     multiple real nodes.
#     """
#     Graph_volume_lattice = build_lattice_graph(d,cycles,volume_lattice)
#     Graph_fault = SimpleGraph(length(fault_nodes))
#     for 

# Recovery Section
function find_data_qubit_z(d::Int64, surface_code_lattice::Dict, x_ancilla_list::Vector{Int64}, 
    fault_edge::Vector{Int64})
    # Fix for the errors consisting of the ghost nodes 
    ancilla_pair::Vector{Int64} = [x_ancilla_list[fault_edge[1]], x_ancilla_list[fault_edge[2]]]
    if isempty(intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[2]]))
        # two faults
        return [intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[1] + d-1]),
        intersect(surface_code_lattice[ancilla_pair[2]], surface_code_lattice[ancilla_pair[2] + 1])]
    else
        # one fault
        return intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[2]])
    end
end

function find_data_qubit_x(d::Int64, surface_code_lattice::Dict, z_ancilla_list::Vector{Int64}, 
    fault_edge::Vector{Int64})
    # Fix for the errors consisting of ghost nodes 
    ancilla_pair::Vector{Int64} = [z_ancilla_list[fault_edge[1]], z_ancilla_list[fault_edge[2]]]
    if isempty(intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[2]]))
        # two faults
        return [intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[1] + d]),
        intersect(surface_code_lattice[ancilla_pair[2]], surface_code_lattice[ancilla_pair[2] + 1])]
    else
        # one fault
        return intersect(surface_code_lattice[ancilla_pair[1]], surface_code_lattice[ancilla_pair[2]])
    end
end

function find_error_qubit(d::Int64, surface_code_lattice::Dict, x_ancilla_list::Vector{Int64}, 
    z_ancilla_list::Vector{Int64}, x_edge_list::Tuple, z_edge_list::Tuple)
    x_error_qubits::Vector{Int64} = [] # qubits where x error occurred 
    z_error_qubits::Vector{Int64} = [] # qubits where z error occurred
    # Assuming the nodes of each edge is in increasing order
    for edge in x_edge_list
        if edge[2] == edge[1] + d*(d+1)
            # Fault on an ancilla qubit: find the corresponding qubit at t=0
            append!(z_error_qubits, edge[2] - floor(Int64, edge[2]/(d*(d+1)))*d*(d+1))
        else
            # Fault on a data qubit: find the corresponding qubit at t=0
            fault_edge::Vector{Int64} = [edge[1] - floor(Int64, edge[1]/(d*(d+1)))*d*(d+1), 
            edge[2] - floor(Int64, edge[2]/(d*(d+1)))*d*(d+1)]
            append!(z_error_qubits, find_data_qubit_z(d, surface_code_lattice, x_ancilla_list,
             fault_edge))
        end
    end

    for edge in z_edge_list
        if edge[2] == edge[1] + d*(d+1)
            # Fault on an ancilla qubit: find the corresponding qubit at t=0
            append!(x_error_qubits, edge[2] - floor(Int64, edge[2]/(d*(d+1)))*d*(d+1))
        else
            # Fault on a data qubit: find the corresponding qubit at t=0
            fault_edge::Vector{Int64} = [edge[1] - floor(Int64, edge[1]/(d*(d+1)))*d*(d+1), 
            edge[2] - floor(Int64, edge[2]/(d*(d+1)))*d*(d+1)]
            append!(x_error_qubits, find_data_qubit_x(d, surface_code_lattice, z_ancilla_list,
             fault_edge))
        end
    end
    return z_error_qubits, x_error_qubits
end
 

# function circuit_x_ancilla!(QS::QuantumState, graph::Dict, x_ancillas::Vector{Int64}, 
#     measurement::Vector{Int64}, pdf::Vector{Float64})
#     for i in x_ancillas
#         apply_h!(QS,i)
#         sampled_gate = samplingDistribution(pdf)
#         gateOperation!(QS,sampled_gate,i)
#         for j = 1:length(graph[i])
#             for k = 1:length(graph[i])
#                 if j == k
#                     apply_cnot!(QS,i,graph[i][j])
#                     sampled_gate = samplingDistribution(pdf)
#                     gateOperation!(QS,sampled_gate,i)
#                     sampled_gate = samplingDistribution(pdf)
#                     gateOperation!(QS,sampled_gate,j)
#                 end
#             end
#         end
#         apply_h!(QS,i)
#         sampled_gate = samplingDistribution(pdf)
#         gateOperation!(QS,sampled_gate,i)
#         measurement[i] = measure!(QS,i)
#     end
# end

# function circuit_z_ancilla!(QS::QuantumState, graph::Dict, z_ancillas::Vector{Int64}, 
#     measurement::Vector{Int64}, pdf::Vector{Float64})
#     for i in z_ancillas
#         for j = 1:length(graph[i])
#             for k = 1:length(graph[i])
#                 if j == k
#                     apply_cnot!(QS,graph[i][j],i)
#                     sampled_gate = samplingDistribution(pdf)
#                     gateOperation!(QS,sampled_gate,i)
#                     sampled_gate = samplingDistribution(pdf)
#                     gateOperation!(QS,sampled_gate,j)
#                 end
#             end
#         end
#         measurement[i] = measure!(QS,i)
#     end
# end

function main(d::Int64)
    # initialize values and get the connections between qubits. 
    total_qubits::Int64 = (2*d-1)^2
    QS = QuantumState(total_qubits)
    connections::Dict = generate_connections(d)
    measurement_values::Vector{Int64} = zeros(Int64, total_qubits)
    

    # generate the different sets of qubits
    data_qubits_list::Vector{Int64} = generate_data_qubits(total_qubits)
    x_ancilla_list::Vector{Int64} = generate_x_ancillas(d, total_qubits)
    z_ancilla_list::Vector{Int64} = generate_z_ancillas(d, total_qubits)
    
    # measurement_circuit!(QS, d, connections, x_ancilla_list, z_ancilla_list)
    measurement_circuit!(QS, d, connections, x_ancilla_list, z_ancilla_list)

    for i in x_ancilla_list
        measurement_values[i] = measure!(QS, i)
    end

    for i in z_ancilla_list
        measurement_values[i] = measure!(QS, i)
    end

    I, J, V = findnz(sparse(QS.tableau))
    df = DataFrame([:I => I, :J => J])
    CSV.write("spmatrix.csv", df)

    # measurement_cycles = [[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
    # [0,1,0,1,0,-1,0,1,0,1,0,1,0,1,0,1,0,-1,0,1,0,1,0,1,0],
    # [0,1,0,1,0,-1,0,1,0,1,0,1,0,1,0,1,0,-1,0,1,0,1,0,1,0],
    # [0,1,0,1,0,-1,0,1,0,1,0,1,0,1,0,1,0,-1,0,1,0,1,0,1,0]]

    # generate_fault_graph(QS, d, connections, x_ancilla_list, z_ancilla_list, 
    # measurement_values)

    # fault_list = generate_fault_nodes(d, 3, measurement_cycles, x_ancilla_list, z_ancilla_list)
    # display(fault_list)

    # gamma::Float64 = 0.05
    # coeff_gates::Vector{Float64} = [(1.0-gamma)/2+sqrt(1.0-gamma)/2, (1.0-gamma)/2-sqrt(1.0-gamma)/2, gamma]
    # prob_distribution::Vector{Float64} = probDistribution(coeff_gates)
    # circuit_z_ancilla!(QS, graph, z_ancilla_list, measurement_values, prob_distribution)
    # for i in data_qubits_list
    #     apply_h!(QS,i)
    #     sampled_gate = samplingDistribution(prob_distribution)
    #     gateOperation!(QS,sampled_gate,i)
    # end
    # circuit_x_ancilla!(QS, graph, x_ancilla_list, measurement_values, prob_distribution)
    # for j in data_qubits_list
    #     apply_h!(QS,j)
    # end
    # return reshape(measurement_values, (5,5))
end

main(3)
