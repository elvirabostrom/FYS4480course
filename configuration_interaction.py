import numpy as np  
from itertools import combinations

"""
CI for pairing model

Claude OPus 5.5 has been used for debugging
"""

def pairingHamiltonian(g, e = 1):
	# initialize pairing model Hamiltonian, doubles and quadruples 
	H = np.ones((6,6)) * ( - 1 / 2 * g)
	H[np.arange(6), np.arange(5, -1, -1)] = 0
	configs = list(combinations([1, 2, 3, 4], 2))  # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
	diag_elem = np.array([sum(2 * e * (p - 1) for p in c) for c in configs]) - g
	np.fill_diagonal(H, diag_elem)
	return H


def pairingCID(g, e = 1):
	# initialize pairing model Hamiltonian, doubles approximation
    H = np.ones((5, 5)) * (-1 / 2 * g)
    H[[1, 2, 3, 4], [4, 3, 2, 1]] = 0 
    configs = list(combinations([1, 2, 3, 4], 2))[:5]  # (1,2), (1,3), (1,4), (2,3), (2,4)
    diag_elem = np.array([sum(2 * e * (p - 1) for p in c) for c in configs]) - g
    np.fill_diagonal(H, diag_elem)
    return H


def diagonalize(pairingStrength, exact = True):
	# diagonalize pairing Hamiltonian for multiple interaction strengths
	# exact = True gives the quadruples and doubles solution
	# exact = False gives only doubles
	if exact:
		dim = 6
	else: 
		dim = 5
	E_corr, E_ref = [], [] 
	energy = np.zeros((dim, len(pairingStrength)))
	eigenfuncs = np.zeros((dim, dim, len(pairingStrength)))
	for i, g in enumerate(pairingStrength):
		if exact:
			H = pairingHamiltonian(g)
		else:
			H = pairingCID(g)
		vals, vecs = np.linalg.eigh(H)
		eigenfuncs[:, :, i] = vecs
		energy[:, i] = vals
		E_ref.append(H[0,0])
		E_corr.append(vals[0] - H[0,0])

	return eigenfuncs, energy, E_corr, E_ref


"""
RUNS 
"""

g = np.linspace(-1,1,25)

# CIDQ RUN FOR COEFFICIENTS
C_FCI, _, _, _ = diagonalize(g)
C_FCI = C_FCI[:,0,:] / C_FCI[0, 0, :] # ground state scaled by reference (intermediate normalization)
np.save("results/FCI_coefficients.npy", C_FCI)

# CID RUN FOR COEFFICIENTS
C_CID, _, _, _ = diagonalize(g, exact = False)
C_CID = C_CID[:,0,:] / C_CID[0, 0, :] # ground state scaled by reference (intermediate normalization)
np.save("results/CID_coefficients.npy", C_CID)

# CIDQ RUN FOR ENERGIES
_, E_FCI, _, _ = diagonalize(g)
E_FCI = E_FCI[0,:] # only ground state
np.save("results/FCI_energy_groundstate.npy", E_FCI)

# CIDQ RUN WITH FINER G
g = np.linspace(-1,1,101)
_, E_FCI, _, _ = diagonalize(g)
E_FCI = E_FCI[0,:] # only ground state
np.save("results/FCI_energy_groundstate2.npy", E_FCI)

# CID RUN WITH FINER G
_, E_CID, _, _ = diagonalize(g, exact = False)
E_CID = E_CID[0,:] # only ground state
np.save("results/CID_energy_groundstate2.npy", E_CID)




