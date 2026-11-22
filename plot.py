import matplotlib.pyplot as plt  
import numpy as np 

"""
Visualize results

All plots have been made with the assistance of Claude Opus 5.5, where typically
a simpler self-made version of the plot is given to the LLM with a prompt to e.g.
"also plot this data which has the same format".
"""


g = np.linspace(-1,1,25) # interaction strengths


"""
PT AMPLITUDES AND CI COEFFICIENTS OVER INTERACTION STRENGTHS G
"""

t = np.load("results/PT_amplitudes.npy") # PT amplitudes
C_FCI = np.load("results/FCI_coefficients.npy")[1:] # FCI coefficients (without reference)
C_CID = np.load("results/CID_coefficients.npy")[1:] # CID coefficients (without reference)

panels = [(0, r"$|1010\rangle$"),
          (1, r"$|1001\rangle$ and $|0110\rangle$"), # degenerated SD's
          (3, r"$|0101\rangle$")]

methods = [
    {"name": "PT",  "value": t,     "color": "yellowgreen",    "ls": "--"},
    {"name": "FCI", "value": C_FCI, "color": "k",              "ls": "-"},
    {"name": "CID", "value": C_CID, "color": "cornflowerblue", "ls": "--"},
]

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(7, 6))
axes = axes.ravel()

for ax, (k, title) in zip(axes, panels):
    ax.set_title(title)
    for m in methods:
        ax.plot(g, m["value"][k], label=m["name"], color=m["color"], linestyle=m["ls"])

# relative difference from FCI
ax = axes[3]
ax.set_title("Relative difference from FCI")
for (k, title), ls in zip(panels, ["-", "--", ":"]):
    ref = np.abs(C_FCI[k])
    for m in methods:
        if m["name"] == "FCI":
            continue
        rel = np.divide(np.abs(m["value"][k] - C_FCI[k]), ref,
                        out=np.full_like(ref, np.nan), where=ref > 1e-12)
        ax.plot(g, rel, color=m["color"], linestyle=ls,
                label=f'{m["name"]}, {title}')
ax.set_ylabel(r"$|C - C^{\mathrm{FCI}}| \,/\, |C^{\mathrm{FCI}}|$")
ax.legend(fontsize="small")

# make it pretty
axes[0].legend()
for ax in axes:
    ax.grid(True)
    ax.set_xlim(-1, 1)
for ax in axes[2:]:
    ax.set_xlabel(r"Interaction strength $g$")
axes[0].set_ylabel("Coefficient value")
axes[2].set_ylabel("Coefficient value")
axes[3].set_ylabel(r"$|C - C^{\mathrm{FCI}}|/|C^{\mathrm{FCI}}|$")
fig.tight_layout()
plt.savefig('figures/coefficients_PT_FCI_CID.pdf', bbox_inches='tight')
#plt.show()

# 4p4h coefficients from FCI
#print("4p4h coefficients from FCI: ", C_FCI[4:])

# 4p4h coefficients from PT second order wave operator (over all g)
t_0011 = np.array(t[0,:]) * np.array(t[3,:]) + np.array(t[1,:]) * np.array(t[2,:])
#print("4p4h coefficients from PT second order: ", t_0011)



"""
PT EXPECTATION IN FIRST ORDER WAVE FUNC COMPARED TO FCI

(normalized wavefunction)
"""

E_PT_first = np.load("results/PT_expec_first_order.npy") # PT expectation values in wave equation to first order, over g
E_FCI = np.load("results/FCI_energy_groundstate.npy") # FCI ground state energy over g

g_marks = [-1, -0.5, 0, 0.5, 1]
idx = [np.argmin(np.abs(g - x)) for x in g_marks]  # index of the g value closest to each mark

fig = plt.figure(figsize=(4, 3))
plt.plot(g, E_PT_first - E_FCI, color = "yellowgreen", marker = "s", markevery = idx)
plt.axhline(0, color = "k", lw = 0.8)
plt.xlabel(r"Interaction strength $g$")
plt.ylabel(r"$E_\mathrm{PT} - E_\mathrm{FCI}$")
plt.xlim(-1, 1)
plt.grid()
fig.tight_layout()
plt.savefig('figures/expec_PTfirst_FCI.pdf', bbox_inches='tight')
plt.show()


"""
THIRD ORDER CORRECTION BY MATRICES (3iv)- save for later???
"""

# H_I = -g/2 * np.array([[2, 1, 1, 1, 1],
#                        [1, 2, 1, 1, 0],
#                        [1, 1, 2, 0, 1],
#                        [1, 1, 0, 2, 1],
#                        [1, 0, 1, 1, 2]])

# R = np.array([[0, 0, 0, 0, 0],
#                      [0, 1/(W_0-W_1), 0, 0, 0],
#                      [0, 0, 1/(W_0-W_2), 0, 0],
#                      [0, 0, 0, 1/(W_0-W_3), 0],
#                      [0, 0, 0, 0, 1/(W_0-W_4)]]) 

# Phi0 = np.array([1, 0, 0, 0, 0])


"""
ENERGY THROUGH THIRD ORDER BY SD NOTATION
"""
g = np.linspace(-1,1,101)
E_ref = 2 - g

E2_SD = np.load("results/energy_second_correction_SD.npy")
E3_SD = np.load("results/energy_third_correction_SD.npy")

E_second_SD = E_ref + E2_SD
E_third_SD = E_second_SD + E3_SD


"""
ENERGY THROUGH THIRD ORDER BY SP NOTATION
"""
E2_SP = np.load("results/energy_second_correction_SP.npy")	
E3_SP = np.load("results/energy_third_correction_SP.npy")

E_second_SP = E_ref + E2_SP
E_third_SP = E_second_SP + E3_SP


"""
Analytical energies through second and third order
"""
E2_analytic = E_ref - 7 * g**2 / 24
E3_analytic = E2_analytic - g**3 / 12

# check if methods agree with analytic
print("second order energy from SD agrees w analytic: ", np.allclose(E_second_SD, E2_analytic, rtol=1e-12, atol=1e-16, equal_nan=False))
print("third order energy from SD agrees w analytic: ", np.allclose(E_third_SD, E3_analytic, rtol=1e-12, atol=1e-16, equal_nan=False))
print("second order energy from SP agrees w analytic: ", np.allclose(E_second_SP, E2_analytic, rtol=1e-12, atol=1e-16, equal_nan=False))
print("third order energy from SP agrees w analytic: ", np.allclose(E_third_SP, E3_analytic, rtol=1e-12, atol=1e-16, equal_nan=False))
# second order energy from SD agrees w analytic:  True
# third order energy from SD agrees w analytic:  True
# second order energy from SP agrees w analytic:  True
# third order energy from SP agrees w analytic:  True
# slay

# check SD and SP if they agree
print("second order correction from SD and SP agrees: ", np.allclose(E2_SD, E2_SP, rtol=1e-12, atol=1e-16, equal_nan=False))
print("third order correction from SD and SP agrees: ", np.allclose(E3_SD, E3_SP, rtol=1e-12, atol=1e-16, equal_nan=False))
# second order correction from SD and SP agrees:  True
# third order correction from SD and SP agrees:  True


"""
ENERGIES THROUGH SECOND AND THIRD ORDER COMPARED TO FCI
"""
E_FCI = np.load("results/FCI_energy_groundstate2.npy")

energies = [
    {"name": "Reference", "value": E_ref,       "color": "grey",           "ls": ":"},
    {"name": "2nd order", "value": E_second_SD, "color": "yellowgreen",    "ls": "--"},
    {"name": "3rd order", "value": E_third_SD,  "color": "cornflowerblue", "ls": "--"},
]

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8, 3))

# energies
ax = axes[0]
ax.plot(g, E_FCI, color="k", label="FCI")
for e in energies:
    ax.plot(g, e["value"], color=e["color"], linestyle=e["ls"], label=e["name"])
ax.set_ylabel(r"Energy $E$")
ax.legend(fontsize="small")

# absolute deviation from FCI, log scale, g = 0 left out
mask = np.abs(g) > 1e-12
ax = axes[1]
for e in energies:
    ax.semilogy(g[mask], np.abs(e["value"] - E_FCI)[mask], color=e["color"], linestyle=e["ls"])
ax.axhline(1e-3, color="k", lw=0.8)
ax.set_ylabel(r"$|E - E_\mathrm{FCI}|$")

for ax in axes:
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"Interaction strength $g$")
fig.tight_layout()
plt.savefig('figures/energy_PT_FCI.pdf', bbox_inches='tight')
plt.show()


"""
HF SPLITTING
"""
E2_HF = np.load("results/energy_second_correction_HF.npy")	
E3_HF = np.load("results/energy_third_correction_HF.npy")

E_second_HF = E_ref + E2_HF
E_third_HF = E_second_HF + E3_HF


# Analytical comparison of E2
E2_analytic = - g**2 / 4 * (2 / (4 + g) + 1 / (2 + g) + 1 / (6 + g))
print("second order correction from HF agrees w analytic: ", np.allclose(E2_HF, E2_analytic, rtol=1e-12, atol=1e-16, equal_nan=False))
# second order correction from HF agrees w analytic:  True

"""
ENERGIES WITH STANDARD AND HF SPLITTING COMPARED TO FCI
"""

energies = [
    {"name": "Reference",     "value": E_ref,       "color": "grey",           "ls": ":"},
    {"name": "2nd order",     "value": E_second_SD, "color": "yellowgreen",    "ls": "--"},
    {"name": "3rd order",     "value": E_third_SD,  "color": "cornflowerblue", "ls": "--"},
    {"name": "2nd order, HF", "value": E_second_HF, "color": "olivedrab", "ls": "-."},
    {"name": "3rd order, HF", "value": E_third_HF,  "color": "navy",      "ls": "-."},
]

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8, 3))

# energies
ax = axes[0]
ax.plot(g, E_FCI, color="k", label="FCI")
for e in energies:
    ax.plot(g, e["value"], color=e["color"], linestyle=e["ls"], label=e["name"])
ax.set_ylabel(r"Energy $E$")
ax.legend(fontsize="small")

# absolute deviation from FCI, log scale, g = 0 left out
mask = np.abs(g) > 1e-12
ax = axes[1]
for e in energies:
    ax.semilogy(g[mask], np.abs(e["value"] - E_FCI)[mask], color=e["color"], linestyle=e["ls"])
ax.axhline(1e-3, color="k", lw=0.8)
ax.set_ylabel(r"$|E - E_\mathrm{FCI}|$")

for ax in axes:
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"Interaction strength $g$")
fig.tight_layout()
plt.savefig('figures/energy_PT_HF_FCI.pdf', bbox_inches='tight')
plt.show()

# energies at selected g values
for x in [-1, -0.5, 0, 0.5, 1]:
    i = np.argmin(np.abs(g - x))
    print(f"g = {x:5.1f}:  ref = {E_ref[i]:.4f},  "
          f"2nd = {E_second_SD[i]:.4f},  3rd = {E_third_SD[i]:.4f},  "
          f"2nd HF = {E_second_HF[i]:.4f},  3rd HF = {E_third_HF[i]:.4f},  "
          f"FCI = {E_FCI[i]:.4f}")



"""
FINAL COMPARISON
"""

# E_FCI is exact
# E_second_SD, E_third_SD, E_second_HF, E_third_HF are RSPT
E_CID = np.load("results/CID_energy_groundstate2.npy") # CI doubles
E_HF = np.load("results/HF_energy.npy") # Hartree Fock
E_X1 = np.load("results/DMRG1_energy.npy") # truncated MPS energy, bond dim 1
E_X2 = np.load("results/DMRG2_energy.npy") # truncated MPS energy, bond dim 2

def print_table_all_methods(g_values=[-1, -0.5, 0, 0.5, 1]):
	# print correlation energy E - E_ref for all methods at selected g
	methods = {
		"exact":     E_FCI,
		"CID":       E_CID,
		"chi=2":     E_X2,
		"chi=1":     E_X1,
		"HF":        E_HF,
		"RSPT2":     E_second_SD,
		"RSPT3":     E_third_SD,
		"RSPT2, HF": E_second_HF,
		"RSPT3, HF": E_third_HF,
	}
	print("All methods, E_corr = E - E_ref")
	for x in g_values:
		i = np.argmin(np.abs(g - x))  # index of the g value closest to x
		print(f"g = {g[i]:.2f}")
		for name, E in methods.items():
			print(f"  E_corr {name:<10} = {E[i] - E_ref[i]: .4f}")

print_table_all_methods()

"""
ALL METHODS COMPARED TO FCI
"""

# first midterm: mean field, CI and MPS
midterm1 = [
    {"name": "HF",             "value": E_HF,  "color": "grey",       "ls": "-",    "marker": None},
    {"name": r"MPS, $\chi=1$", "value": E_X1,  "color": "k",          "ls": "none", "marker": "o"},
    {"name": "CID",            "value": E_CID, "color": "darkorange", "ls": "-",    "marker": None},
    {"name": r"MPS, $\chi=2$", "value": E_X2,  "color": "crimson",    "ls": ":",    "marker": None},
]
# second midterm: perturbation theory
midterm2 = [
    {"name": "RSPT2",     "value": E_second_SD, "color": "yellowgreen",    "ls": "--", "marker": None},
    {"name": "RSPT3",     "value": E_third_SD,  "color": "cornflowerblue", "ls": "--", "marker": None},
    {"name": "RSPT2, HF", "value": E_second_HF, "color": "olivedrab",      "ls": "-.", "marker": None},
    {"name": "RSPT3, HF", "value": E_third_HF,  "color": "navy",           "ls": "-.", "marker": None},
]

g_marks = np.linspace(-1, 1, 9)
idx = [np.argmin(np.abs(g - x)) for x in g_marks]   # marker positions
mask = np.abs(g) > 1e-12                            # g = 0 is left out on the log scale

def draw(ax, m, log=False):
    # plot one method, either the energy or the absolute deviation from FCI
    y = m["value"]
    if log:
        y = np.where(mask, np.abs(y - E_FCI), np.nan)   # NaN at g = 0
    ax.plot(g, y, color=m["color"], linestyle=m["ls"], marker=m["marker"],
            markevery=idx, markersize=4, label=m["name"])
    if log:
        ax.set_yscale("log")

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)
ax_E = fig.add_subplot(gs[0, :])
ax_1 = fig.add_subplot(gs[1, 0])
ax_2 = fig.add_subplot(gs[1, 1], sharey=ax_1)

# energies, all methods
ax_E.plot(g, E_FCI, color="k", label="FCI")
for m in midterm1 + midterm2:
    draw(ax_E, m)
ax_E.set_ylabel(r"Energy $E$")
ax_E.legend(fontsize="small", ncol=3)

# absolute deviation from FCI, first midterm
ax_1.set_title("First midterm")
for m in midterm1:
    draw(ax_1, m, log=True)
ax_1.set_ylabel(r"$|E - E_\mathrm{FCI}|$")

# absolute deviation from FCI, second midterm
ax_2.set_title("Perturbation theory")
for m in midterm2:
    draw(ax_2, m, log=True)

for ax in [ax_1, ax_2]:
    ax.axhline(1e-3, color="k", lw=0.8)
    ax.legend(fontsize="small")
for ax in [ax_E, ax_1, ax_2]:
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"Interaction strength $g$")
fig.tight_layout()
plt.savefig('figures/energy_all_methods.pdf', bbox_inches='tight')
plt.show()


