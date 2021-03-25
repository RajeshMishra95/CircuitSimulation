using SparseArrays
using RowEchelon
using DataFrames
using CSV
using LinearAlgebra

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
        return 1
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
            return 1
        elseif QS.tableau[Int64(prob_measure[3]),2*QS.qubits+1] == 1
            return -1
        end
    end
end

function find_logical_state(QS::Array{Int64,2}, d::Int64)
    """
    This function calculates the logical state of the surface code.
    """
    total_qubits::Int64 = (2*d-1)^2
    anti_comm_stabilizer::Vector{Int64} = []
    # Initialize the Z-logical operator
    z_logical::Vector{Int64} = zeros(Int64, 2*total_qubits + 1)
    for i = 1:d
        z_logical[total_qubits + 1+2*(i-1)] = 1
    end

    # Find the destabilizers that do not commute with the Z-logical operator
    for i = 1:total_qubits
        # if length(findall(x -> x == 1, 
        #     z_logical[1:2*total_qubits].*QS[i,1:2*total_qubits]))%2 == 1
        #     append!(anti_comm_stabilizer, i)
        # end
        display(length(findall(x -> x == 1,z_logical[total_qubits+1:2*total_qubits].*QS[i,1:total_qubits])))
    end

    # # Rowsum the corresponding stabilizers with the last row 
    # QS.tableau = vcat(QS.tableau, zeros(Int64,1,2*QS.qubits+1))
    # for j in anti_comm_stabilizer
    #     rowsum!(QS, 2*QS.qubits+1, j+QS.qubits) 
    # end
    # return QS.tableau[end,end]
end

function check_state()
    m = DataFrame(CSV.File("stabilizer.csv"))
    V = ones(Int64, 129)
    J = m.J
    I = m.I
    mat = Matrix(sparse(I, J, V))
    find_logical_state(mat, 3)
end