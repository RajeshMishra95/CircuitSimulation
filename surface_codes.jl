using LinearAlgebra
using DelimitedFiles
using Statistics

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

function generate_graph(n::Int64)
    d::Int64 = n
    dim::Int64 = 2*d - 1
    total_qubits::Int64 = dim*dim
    graph_dict = Dict{Int64,Array{Int64,1}}()
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
        push!(graph_dict, i => list_connections)
    end
    return graph_dict
end

function gateOperation!(QS::QuantumState, gate::Int64, qubit::Int64)
    if gate == 2
        apply_z!(QS, qubit)
    elseif gate == 3
        apply_Rz!(QS, qubit)
    end
end

function circuit_x_ancilla!(QS::QuantumState, graph::Dict, x_ancillas::Vector{Int64}, 
    measurement::Vector{Int64}, pdf::Vector{Float64})
    for i in x_ancillas
        apply_h!(QS,i)
        sampled_gate = samplingDistribution(pdf)
        gateOperation!(QS,sampled_gate,i)
        for j = 1:length(graph[i])
            for k = 1:length(graph[i])
                if j == k
                    apply_cnot!(QS,i,graph[i][j])
                    sampled_gate = samplingDistribution(pdf)
                    gateOperation!(QS,sampled_gate,i)
                    sampled_gate = samplingDistribution(pdf)
                    gateOperation!(QS,sampled_gate,j)
                end
            end
        end
        apply_h!(QS,i)
        sampled_gate = samplingDistribution(pdf)
        gateOperation!(QS,sampled_gate,i)
        measurement[i] = measure!(QS,i)
    end
end

function circuit_z_ancilla!(QS::QuantumState, graph::Dict, z_ancillas::Vector{Int64}, 
    measurement::Vector{Int64}, pdf::Vector{Float64})
    for i in z_ancillas
        for j = 1:length(graph[i])
            for k = 1:length(graph[i])
                if j == k
                    apply_cnot!(QS,graph[i][j],i)
                    sampled_gate = samplingDistribution(pdf)
                    gateOperation!(QS,sampled_gate,i)
                    sampled_gate = samplingDistribution(pdf)
                    gateOperation!(QS,sampled_gate,j)
                end
            end
        end
        measurement[i] = measure!(QS,i)
    end
end

function main()
    d::Int64 = 3
    total_qubits::Int64 = (2*d-1)^2
    QS = QuantumState(total_qubits)
    graph::Dict = generate_graph(d)
    measurement_values::Vector{Int64} = zeros(Int64, total_qubits)
    data_qubits_list::Vector{Int64} = [1,3,5,7,9,11,13,15,17,19,21,23,25]
    x_ancilla_list::Vector{Int64} = [2,4,12,14,22,24]
    z_ancilla_list::Vector{Int64} = [6,8,10,16,18,20]
    gamma::Float64 = 0.01
    coeff_gates::Vector{Float64} = [(1.0-gamma)/2+sqrt(1.0-gamma)/2, (1.0-gamma)/2-sqrt(1.0-gamma)/2, gamma]
    prob_distribution::Vector{Float64} = probDistribution(coeff_gates)
    circuit_z_ancilla!(QS, graph, z_ancilla_list, measurement_values, prob_distribution)
    for i in data_qubits_list
        apply_h!(QS,i)
        sampled_gate = samplingDistribution(prob_distribution)
        gateOperation!(QS,sampled_gate,i)
    end
    circuit_x_ancilla!(QS, graph, x_ancilla_list, measurement_values, prob_distribution)
    for j in data_qubits_list
        apply_h!(QS,j)
    end
    return reshape(measurement_values, (5,5))
end

