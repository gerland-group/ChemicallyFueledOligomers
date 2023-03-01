#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import pandas as pd

# import own modules
import complexes
import datareader
import evaluator
import reactions
import ratekernel

# read experimental data
times_exp, map_oligos_exp, c_oligos_exp, c_oligos_exp_err, c_EDC_exp, c_EDC_exp_err = \
    datareader.read_experimental_data_T25_EDC10()

# set system parameters
# maximum length of considered oligomer
n_max = 7
# length of template
Lt = 0
# maximum number of oligomers hybridized to template
degree_total = 0
# maximum number of O-Acylisourea oligomers hybridized to template
degree_O = 0
# maximum number of N-Acylisourea oligomers hybridized to template
degree_N = 0

# initialize table listing all chemical compounds of interest
table_sol = complexes.generate_table_full_complexity(n_max, Lt, \
    degree_total, degree_O, degree_N)

# initialize table that lists all reaction rate constants in the rate constant kernel
table_rate_constants = ratekernel.list_rate_constants()

# list all chemical reactions in solution
acts_sol, acts_sol_humanreadable = reactions.list_activations_solution(n_max, table_sol)
ligs_sol, ligs_sol_humanreadable = reactions.list_ligations_solution(n_max, table_sol)
losses_sol, losses_sol_humanreadable = reactions.list_losses_solution(n_max, table_sol)
hydros_sol, hydros_sol_humanreadable = reactions.list_hydrolysis_solution(n_max, table_sol)
cuts_sol, cuts_sol_humanreadable = reactions.list_cleavages_solution(n_max, table_sol)

# initial concentration
c_full_initial = np.zeros(len(table_sol)+1)
# useable monomer concentration
c_full_initial[table_sol['1,_,0']] = 25.
# EDC concentration
c_full_initial[-1] = 10.

# read reaction rate constants obtained via curve fits
df = pd.read_excel('./rate_constants_variable_klig.xlsx')
ks_multi = np.zeros((len(df), len(table_rate_constants)))
for name in list(df.columns):
    for i in range(len(df)):
        ks_multi[i][table_rate_constants[name[1:]]] = df[name][i]

# compute goodness of fit for all read reaction rate constants
chi_multi = np.zeros(len(df))

# compute time evolution
for i in range(len(ks_multi)):
    print(f"set of rate constants {i} out of {len(ks_multi)}")
    _, _, _, _, chi = evaluator.model_solution_kernel(\
        np.log2(ks_multi[i]), c_full_initial, times_exp, c_oligos_exp, map_oligos_exp, \
        c_EDC_exp, acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, table_sol, \
        table_rate_constants, n_max, 0., True, False, False)
    chi_multi[i] = chi

# generate plot
fig = plt.figure(figsize=(9,9), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
grids = [gs[0,0:2], gs[0,2:], gs[1,0:2], gs[1,2:], gs[2,0:2]]

# kact is not sensitive to choice of klig
names_act = [r'$l=1$', r'$l=2$', r'$l=3$', r'$l=4$', r'$l\geq5$']
ax = fig.add_subplot(grids[0])
for i_react, react in enumerate(['act_1', 'act_2', 'act_3', 'act_4', 'act_rest']):
    ax.plot(ks_multi[:,table_rate_constants['lig']], ks_multi[:,table_rate_constants[react]], \
        label = names_act[i_react])
ax.set_xlabel(r'$k_{lig}$ (mM$^{-1}$h$^{-1}$)')
ax.set_ylabel(r'$k_{act}$ (mM$^{-1}$h$^{-1}$)')
ax.legend()
ax.grid(alpha=0.3)

# kcut is not sensitive to choice of klig
names_cut = [r'$l=2$', r'$l=3$', r'$l=4$', r'$l\geq5$']
ax = fig.add_subplot(grids[1])
for i_react, react in enumerate(['cut_2', 'cut_3', 'cut_4', 'cut_rest']):
    ax.plot(ks_multi[:,table_rate_constants['lig']], ks_multi[:,table_rate_constants[react]], \
        label = names_cut[i_react])
ax.set_xlabel(r'$k_{lig}$ (mM$^{-1}$h$^{-1}$)')
ax.set_ylabel(r'$k_{cut}$ (h$^{-1}$)')
ax.legend()
ax.grid(alpha=0.3)

# klig is linearly correlated with kloss
ax = fig.add_subplot(grids[2])
ax.plot(ks_multi[:,table_rate_constants['lig']], ks_multi[:,table_rate_constants['loss']])
ax.set_xlabel(r'$k_{lig}$ (mM$^{-1}$h$^{-1}$)')
ax.set_ylabel(r'$k_{loss}$ (h$^{-1}$)')
ax.grid(alpha=0.3)

# klig is linearly correlated with khydro
ax = fig.add_subplot(grids[3])
ax.plot(ks_multi[:,table_rate_constants['lig']], ks_multi[:,table_rate_constants['hydro']])
#ax.set_xlim([5e3, 1e5])
ax.set_xlabel(r'$k_{lig}$ (mM$^{-1}$h$^{-1}$)')
ax.set_ylabel(r'$k_{hydro}$ (h$^{-1}$)')
ax.grid(alpha=0.3)

# goodness of curve fit (sum of squared residuals) relatively independent of klig
ax = fig.add_subplot(grids[4])
ax.plot(ks_multi[:,table_rate_constants['lig']], chi_multi)
ax.set_xlabel(r'$k_{lig}$ (mM$^{-1}$h$^{-1}$)')
ax.set_ylabel(r'weighted sum of squared residuals $\chi$')
ax.grid(alpha=0.3)

plt.savefig('./dependence_on_klig.pdf')

plt.show()