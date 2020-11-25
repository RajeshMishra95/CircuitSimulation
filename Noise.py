import CHP

def apply_noise(QS, noise, qubit):
    if noise == 2:
        QS.apply_x(qubit)
    elif noise == 3:
        QS.apply_y(qubit)
    elif noise == 4:
        QS.apply_z(qubit)
    elif noise == 5:
        QS.apply_s(qubit)
    elif noise == 6:
        QS.apply_Rz(qubit)
