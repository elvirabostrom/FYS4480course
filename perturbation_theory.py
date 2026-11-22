import numpy as np  
from itertools import combinations

"""
PT for pairing model

Claude OPus 5.5 has been used for debugging
"""

def PT_amplitudes_first_order(pairingStrength, e=1):
	# compute amplitudes for 2p2h determinants for multiple interaction strengths
	# first order contribution to the wave function
    configs = list(combinations([1, 2, 3, 4], 2))[:5]  # (1,2), (1,3), (1,4), (2,3), (2,4)
    pair_energies = np.array([sum(2 * e * (p - 1) for p in c) for c in configs])
    diffs = pair_energies[0] - pair_energies[1:]   # D = W_0 - W_m
    t = np.zeros((len(diffs), len(pairingStrength)))
    for j, g in enumerate(pairingStrength):
    		t[:, j] = - g / (2 * diffs)
    return t


def PT_expec_first_order(pairingStrength):
	# compute expectation value in wave equation to first order
	# SCALED TO NORM 1
	# contains contributions from E^1, E^2 and E^3
	t = PT_amplitudes_first_order(pairingStrength)
	E = np.zeros((len(pairingStrength)))
	for i, g in enumerate(pairingStrength):
		E_ref = 2 - g
		norm = np.sum(t[:,i]**2)
		E2 = np.sum(-g/2 * t[:,i])
		V = -g/2 * np.array([[2, 1, 1, 0],
                             [1, 2, 0, 1],
                             [1, 0, 2, 1],
                             [0, 1, 1, 2]])
		E3 = t[:,i] @ V @ t[:,i] + g * norm
		E[i] = E_ref + (E2 + E3) / (1 + norm)
	return E


def SD_matrices():
	# precalculate matrices for SD calculation of energy
	w = 0.5 * np.ones(4)
	W = 0.5 * np.array([[0, 1, 1, 0],
                        [1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [0, 1, 1, 0]])
	R = -np.diag([1 / 2, 1 / 4, 1 / 4, 1 / 6])
	return w, W, R


def energy_third_order_SD(pairingStrength):
	# compute energy through third order from slater determinant basis notation, over g
	E2 = np.zeros(len(pairingStrength))
	E3 = np.zeros(len(pairingStrength))
	w, W, R = SD_matrices()
	for i, g in enumerate(pairingStrength):
		wi = - g * w
		Wi = - g * W
		E2[i] = wi @ R @ wi
		E3[i] = wi @ R @ Wi @ R @ wi
	return E2, E3


def SP_matrices():
	# precalculate matrices for SP calculation of energy
	T = np.array([[1 / 8, 1 / 4],
	              [1 / 12, 1 / 8]])
	V = 0.5 * np.ones((2, 2))
	return T, V


def energy_third_order_SP(pairingStrength):
	# compute energy through third order from single particle basis notation, over g
	E2 = np.zeros(len(pairingStrength))
	E3 = np.zeros(len(pairingStrength))
	T, V = SP_matrices()
	t_all = PT_amplitudes_first_order(pairingStrength)
	for i, g in enumerate(pairingStrength):
		Ti = g * T
		Vi = - g * V
		VTi = Vi @ Ti
		f_h = -g / 2 # one-body vertex, diagonal in the pairing model

		E2[i] = np.trace(VTi) # diagram 1 (diagram 2 is zero in pairing model)
		D4 = np.trace(Ti.T @ VTi) # pp ladder
		D5 = np.trace(Ti @ Vi @ Ti.T) # hh ladder
		D8 = 2 * (- f_h) * np.sum(t_all[:, i]**2) # one-body vertex
		E3[i] = D4 + D5 + D8
	return E2, E3


def energy_third_order_HF(pairingStrength):
	# compute energy through third order with HF splitting of matrix, over g
	E2 = np.zeros(len(pairingStrength))
	E3 = np.zeros(len(pairingStrength))
	w = 0.5 * np.ones(4)
	W = 0.5 * np.array([[0, 1, 1, 0],
		                [1, 0, 0, 1],
		                [1, 0, 0, 1],
		                [0, 1, 1, 0]])
	for i, g in enumerate(pairingStrength):
		R = -np.diag([1 / (2 + g), 1 / (4 + g), 1 / (4 + g), 1 / (6 + g)])
		wi = - g * w
		Wi = - g * W
		E2[i] = wi @ R @ wi
		E3[i] = wi @ R @ (Wi - g * np.eye(4)) @ R @ wi 
	return E2, E3

"""
RUNS 
"""

g = np.linspace(-1,1,25)

# RUN FOR AMPLITUDES
t = PT_amplitudes_first_order(g)
np.save("results/PT_amplitudes.npy", t)


# RUN FOR EXPECTATION VALUES
E = PT_expec_first_order(g)
np.save("results/PT_expec_first_order.npy", E)


# RUN FOR ENERGY TO THIRD ORDER WITH SD NOTATION
g = np.linspace(-1,1,101)

E2, E3 = energy_third_order_SD(g)

np.save("results/energy_second_correction_SD.npy", E2)	
np.save("results/energy_third_correction_SD.npy", E3)	


# RUN FOR ENERGY TO THIRD ORDER WITH SP NOTATION
E2, E3 = energy_third_order_SP(g)

np.save("results/energy_second_correction_SP.npy", E2)	
np.save("results/energy_third_correction_SP.npy", E3)


# RUN FOR HARTREE FOCK SPLITTING
E2, E3 = energy_third_order_HF(g)

np.save("results/energy_second_correction_HF.npy", E2)	
np.save("results/energy_third_correction_HF.npy", E3)








