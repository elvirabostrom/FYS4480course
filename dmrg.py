import numpy as np  
from itertools import combinations

"""
Energy from MPS for pairing model

Claude OPus 5.5 has been used for debugging and for writing some functions
"""

def pairingHamiltonian(g, e = 1):
	# initialize pairing model Hamiltonian, doubles and quadruples 
	H = np.ones((6,6)) * ( - 1 / 2 * g)
	H[np.arange(6), np.arange(5, -1, -1)] = 0
	configs = list(combinations([1, 2, 3, 4], 2))  # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
	diag_elem = np.array([sum(2 * e * (p - 1) for p in c) for c in configs]) - g
	np.fill_diagonal(H, diag_elem)
	return H

def diagonalize(pairingStrength):
	# diagonalize pairing Hamiltonian for multiple interaction strengths
	dim = 6
	E_corr, E_ref = [], [] 
	energy = np.zeros((dim, len(pairingStrength)))
	eigenfuncs = np.zeros((dim, dim, len(pairingStrength)))
	for i, g in enumerate(pairingStrength):
		H = pairingHamiltonian(g)
		vals, vecs = np.linalg.eigh(H)
		eigenfuncs[:, :, i] = vecs
		energy[:, i] = vals
		E_ref.append(H[0,0])
		E_corr.append(vals[0] - H[0,0])

	return eigenfuncs, energy, E_corr, E_ref


def entropy(s):
	# Claude
	# von Neumann entropy from the singular values of a cut
	w = s[s > 1e-12]**2
	return -np.sum(w * np.log(w))


def computeEntropy(pairingStrength):
	# entropy of the FCI ground state for the three cuts
	S1, S2, S3 = [], [], []
	eigenfuncs, energy, E_corr, E_ref = diagonalize(pairingStrength)
	for i, g in enumerate(pairingStrength):
		full = np.zeros(16); full[[12, 10, 9, 6, 5, 3]] = eigenfuncs[:, 0, i] #Claude
		C1 = full.reshape(2, 8) # first cut
		C2 = full.reshape(4, 4) # middle cut
		C3 = full.reshape(8, 2) # last cut
		_, l1, _ = np.linalg.svd(C1)
		_, l2, _ = np.linalg.svd(C2)
		_, l3, _ = np.linalg.svd(C3)
		S1.append(entropy(l1))
		S2.append(entropy(l2))
		S3.append(entropy(l3))
	return S1, S2, S3


def truncate(psi, bond_dim):
	# truncate the state to bond dimension bond_dim by SVD at the middle cut
	idx = [12, 10, 9, 6, 5, 3] # Claude fixing binary
	full = np.zeros(16); full[idx] = psi
	C2 = full.reshape(4,4)
	U, s, V = np.linalg.svd(C2)
	weight = np.sum(s[bond_dim:]**2)
	C2 = U[:, :bond_dim] @ np.diag(s[:bond_dim]) @ V[:bond_dim]
	psi_trunc = C2.flatten()[idx] 
	psi_trunc /= np.linalg.norm(psi_trunc)
	return psi_trunc, weight


def truncatedEnergies(pairingStrength, bond_dim):
	# energy psi^T H psi of the truncated ground state for each g
	eigenfuncs, _, _, _ = diagonalize(pairingStrength)
	E, w = [], []
	for i, g in enumerate(pairingStrength):
		psi_trunc, weight = truncate(eigenfuncs[:, 0, i], bond_dim)
		E.append(psi_trunc @ pairingHamiltonian(g) @ psi_trunc)
		w.append(weight)
	return E, w


def getWp(p, g, xi=1):
	# MPO tensor for level p
	W = np.zeros((4, 4, 2, 2))
	I  = np.eye(2)
	Pp = np.array([[0, 0], [1, 0]])
	Pm = Pp.T
	n  = np.diag([0, 1])
	eps = 2 * xi * (p - 1)
	h = eps * n - g / 2 * Pp @ Pm 
	W[0,0] = I
	W[1,1] = I
	W[2,2] = I
	W[3,3] = I
	W[0, 1] = -g / 2 * Pp
	W[0, 2] = -g / 2 * Pm;  W[0, 3] = h
	W[1, 3] = Pm
	W[2, 3] = Pp
	return W


def MPO(g):
	# full 16x16 Hamiltonian from the MPO, to check against pairingHamiltonian
    L = np.array([1, 0, 0, 0])
    R = np.array([0, 0, 0, 1])
    W = [getWp(p, g) for p in range(1, 5)]
    H = np.einsum('a,abij,bckl,cdmn,deop,e->ikmojlnp', L, *W, R)
    return H.reshape(16, 16)


def MPS(full):
	# split a 16-dim state into four MPS tensors by successive SVD's
	MPS = []
	rest = full.reshape(1, -1)
	for site in range(3):
		r = rest.shape[0]
		U, s, V = np.linalg.svd(rest.reshape(r * 2, -1), full_matrices=False) # reduced svd
		k = np.sum(s > 1e-12) # keep non zero
		MPS.append(U[:, :k].reshape(r, 2, k))
		rest = np.diag(s[:k]) @ V[:k] 
	MPS.append(rest.reshape(rest.shape[0], 2, 1))
	return MPS


def contract(MPS, g):
	# energy <MPS|MPO|MPS>
	# Claude
    L = np.array([1, 0, 0, 0])
    R = np.array([0, 0, 0, 1])
    W = [getWp(p, g) for p in range(1, 5)]
    return np.einsum('a,abij,bckl,cdmn,deop,e,qir,rks,smt,tou,QjR,RlS,SnT,TpU->', L, *W, R, *MPS, *MPS, optimize=True)


def MPS_MPO_energies(bond_dim, pairingStrength):
	# energy of the truncated ground state as <MPS|MPO|MPS> for each g
	idx = [12, 10, 9, 6, 5, 3]  # Claude fixing binary
	E = []
	eigenfuncs, _, _, _ = diagonalize(pairingStrength)
	for i, g in enumerate(pairingStrength):
		psi_trunc, _ = truncate(eigenfuncs[:, 0, i], bond_dim)
		full = np.zeros(16); full[idx] = psi_trunc
		E.append(contract(MPS(full), g))
	return E


# RUN FOR ENERGIES with bond dims X = 1 and X = 2

g = np.linspace(-1,1,101)
E_X1 = MPS_MPO_energies(1, g)
E_X2 = MPS_MPO_energies(2, g)

np.save("results/DMRG1_energy.npy", E_X1)
np.save("results/DMRG2_energy.npy", E_X2)



