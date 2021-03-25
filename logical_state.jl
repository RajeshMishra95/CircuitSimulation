function find_logical_state(QS::QuantumState, d::Int64)
    """
    This function calculates the logical state of the surface code.
    """
    total_qubits::Int64 = (2*d-1)^2
    anti_comm_stabilizer::Vector{Int64} = []
    # Initialize the X-logical operator
    x_logical::Vector{Int64} = zeros(Int64, 2*total_qubits + 1)
    for i = 1:d
        x_logical[QS.qubits + 1+2*(2*d-1)*(i-1)] = 1
    end

    # Find the destabilizers that do not commute with the X-logical operator
    for i = 1:total_qubits
        if length(findall(x -> x == 1, 
            x_logical[1:2*total_qubits].*QS.tableau[i,1:2*total_qubits]))%2 == 1
            append!(anti_comm_stabilizer, i)
        end
    end
    # Rowsum the corresponding stabilizers with X-logical operator
    QS.tableau = vcat(QS.tableau, zeros(Int64,1,2*QS.qubits+1))
    for j in anti_comm_stabilizer
        rowsum!(QS, 2*QS.qubits+1, j+QS.qubits) 
    end
    return QS.tableau[end,end]
end