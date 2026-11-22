import numpy as np  

"""
Hartree Fock for pairing model

Claude OPus 5.5 has been used for debugging
"""

def initialize_HF():
	# random guess for first iteration
	C_occ = np.linalg.qr(np.random.rand(4, 2))[0].T # orthonormal init
	f = np.zeros((4, 4))
	return C_occ, f


def fock_matrix(C_occ, f, g, xi):
	# update fock matrix
	for i in range(4):
		e = xi * i
		for j in range(4):
			density_mats = C_occ[0][i] * C_occ[0][j] + C_occ[1][i] * C_occ[1][j]
			f[i, j] = - g / 2 * density_mats
			if i == j:
				f[i, j] += e
	return f


def solve_HF(g, tol = 1e-12, xi = 1):
	# solve selfconsistenly for HF energy and wave function, given interaction strength g
	C_occ, f = initialize_HF()
	old = np.zeros(4)
	for it in range(10000):
		f = fock_matrix(C_occ, f, g, xi) # update fock matrix
		vals, vecs = np.linalg.eigh(f) # diagonalize
		C_occ = vecs[:, :2].T # update coeffs for next iteration         
		if np.max(np.abs(vals - old)) < tol:
			break
		old = vals
	SD = vecs[:, :2]
	rho = C_occ.T @ C_occ # rho[l, l'] = sum_j C[j,l] C[j,l']
	E_HF = 2 * np.sum(xi * np.arange(4) * np.diag(rho)) - g / 2 * np.sum(rho**2) # comment that Claude did this from the previuos

	return E_HF, SD



# RUN FOR ENERGIES

pairingStrength = np.linspace(-1,1,101)
E_HF = np.zeros(len(pairingStrength))
for i, g in enumerate(pairingStrength):
	E_HF[i], _ = solve_HF(g)

np.save("results/HF_energy.npy", E_HF)





