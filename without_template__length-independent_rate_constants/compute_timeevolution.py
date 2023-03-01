#!/bin/env python3

import numpy as np
import sys
sys.path.insert(0, '../src')

# import own modules
import complexes
import reactions
import datareader
import evaluator

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

# list all chemical reactions in solution
acts_sol, acts_sol_humanreadable = reactions.list_activations_solution(n_max, table_sol)
ligs_sol, ligs_sol_humanreadable = reactions.list_ligations_solution(n_max, table_sol)
losses_sol, losses_sol_humanreadable = reactions.list_losses_solution(n_max, table_sol)
hydros_sol, hydros_sol_humanreadable = reactions.list_hydrolysis_solution(n_max, table_sol)
cuts_sol, cuts_sol_humanreadable = reactions.list_cleavages_solution(n_max, table_sol)

# initial concentration
c_full_initial = np.zeros(len(table_sol)+1)
# monomer concentration
c_full_initial[table_sol['1,_,0']] = 25.
# EDC concentration
c_full_initial[-1] = 10.

# read reaction rate constant obtained via curve fit
ks = np.loadtxt('./rate_constants.txt')

# compute the time-evolution
times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, res = \
    evaluator.model_solution_scalar(np.log2(ks), c_full_initial, times_exp, \
    c_oligos_exp, map_oligos_exp, c_EDC_exp, acts_sol, ligs_sol, losses_sol, \
    hydros_sol, cuts_sol, table_sol, n_max, True, False)

# plot the time-evolution
evaluator.plot_trajectories_solution(times_exp, c_oligos_exp, map_oligos_exp, c_oligos_exp_err, \
    c_EDC_exp, c_EDC_exp_err, times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, \
    True, "./timeevolution.pdf")
