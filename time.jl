using LinearAlgebra
using DelimitedFiles
using Statistics

mutable struct QuantumState
    tableau::Matrix
    qubits::Int64
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
        return [1,0]
    elseif prob_measure == [0,1]
        return [0,1]
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
            return [1,0]
        elseif QS.tableau[Int64(prob_measure[3]),2*QS.qubits+1] == 1
            return [0,1]
        end
    end
end

function gateOperation!(QS::QuantumState, gate::Union{Char,AbstractString}, qubit::Int64)
    if gate == 'H'
        apply_h!(QS, qubit)
    elseif gate == 'S'
        apply_s!(QS, qubit)
    elseif gate == 'X'
        apply_x!(QS, qubit)
    elseif gate == 'Y'
        apply_y!(QS, qubit)
    elseif gate == 'Z'
        apply_z!(QS, qubit)
    elseif gate == "Rz"
        apply_Rz!(QS, qubit)
    elseif gate == "CNOT"
        apply_cnot!(QS, qubit)
    end
end

function stabgateOperation!(QS::QuantumState, coeff_gates::Array{Float64,1}, prob_gates::Array{Float64,1}, gates::Array{Any,1}, qubit::Int64, w::Float64)
    i = samplingDistribution(prob_gates)
    w *= coeff_gates[i]/prob_gates[i]
    gateOperation!(QS, gates[i], qubit)
    return w
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

#=
function sampledValues(num_qubits::Int64, num_runs::Int64, probabilities::Array{Float64,1})
    gates_applied = Dict()
    pdf = probDistribution(probabilities)
    for i = 1:num_qubits
        qubit_gates = zeros(Int,0)
        for j = 1:num_runs
            append!(qubit_gates, samplingDistribution(pdf))
        end
        push!(gates_applied, i => qubit_gates)
    end
    return gates_applied
end
=#
# time_taken = zeros(0)

function coeff(gamma::Float64)
    coeff_gates = [(1.0-gamma)/2+sqrt(1.0-gamma)/2, (1.0-gamma)/2-sqrt(1.0-gamma)/2, gamma]
#    coeff_gates = [(1.0 + cos(gamma) - sin(gamma))/2, (1 - sin(gamma) - cos(gamma))/2, sin(gamma)]
    return coeff_gates
end

#=
function amplitude_damping(quantum_state, gate_sequence, num_runs, num_qubits)
    for i = 1:num_runs
        for j = 1:num_qubits
            if gate_sequence[j][i] == 2
                quantum_state = apply_z(quantum_state, j, num_qubits)
            elseif gate_sequence[j][i] == 3
                quantum_state = apply_Rz(quantum_state, j, num_qubits)
            end
        end
    end
    return quantum_state
end
=#

function expectation_value!(QS::QuantumState, coeff_gates::Array{Float64,1}, gates::Array{Any,1}, num_runs::Int64, num_gates::Int64)
    prob_gates = probDistribution(coeff_gates)
    exp_value = zeros(Float64,QS.qubits)
    for i = 1:num_runs
        w = zeros(Float64,QS.qubits)
        for item = 1:QS.qubits
            w[item] = 1.0
        end
        rho_star = deepcopy(QS)
        for n = 1:num_gates
            for j = 1:rho_star.qubits
                w[j] = stabgateOperation!(rho_star, coeff_gates, prob_gates, gates, j, w[j])
            end
        end
        
#        apply_s!(rho_star,1)
#        apply_z!(rho_star,1)
#        apply_h!(rho_star,1)

        for k = 1:rho_star.qubits
            prob_outcome = [0,0]
            prob_outcome = measure!(rho_star, k)
            exp_value[k] += w[k]*prob_outcome[1]/num_runs
        end
    end
    return exp_value
end

#=
for i = 1:100
    num_qubits = i
    tableau = hcat(Matrix{Int}(I,2*num_qubits,2*num_qubits),zeros(Int,2*num_qubits,1))

    td_S = QuantumState(tableau)
    
    gate_sequence = sampledValues(num_qubits, 100000, coeff(0.2))

    t1 = time_ns()
    QS.tableau = amplitude_damping(QS.tableau, gate_sequence, 100000, num_qubits) 
    t2 = time_ns()
    println(i)
    
    append!(time_taken,((t2-t1)/1.0e9))
end

writedlm("time_measure2.csv", time_taken,',')
=#

std_dev_as = []

for i = 1:2:100
    num_qubits = 1
    tableau = hcat(Matrix{Int64}(I,2*num_qubits,2*num_qubits),zeros(Int64,2*num_qubits,1))

    QS = QuantumState(tableau,num_qubits)

    gates = ['I','Z',"Rz"]    
    coeff_gates = coeff(0.5)
    

#    std_dev = 1.0
#    k = 0
#    while std_dev > 0.01
    list_values = []
#        k += 1
    for j = 1:100
        append!(list_values, expectation_value!(QS, coeff_gates, gates, 1000, i))
    end
    append!(std_dev_as, Statistics.std(list_values))
#    end
#    append!(std_dev_ap, k*i)
    println(i)
end

writedlm("std_dev_ad2.csv", std_dev_as,',')
#=
    exp_value = []
    for k = 1:100
        coeff_gates = coeff(pi/4)
        append!(exp_value, expectation_value!(QS, coeff_gates, gates, 5000, i))
    end
    append!(std_dev_t_gate, Statistics.std(exp_value))
    println(i)
end

writedlm("std_dev_data1.csv", std_dev_t_gate, ',')
=# 
