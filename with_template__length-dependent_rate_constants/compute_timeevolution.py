#!/env/bin python3.9

import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, '../src')

# import own modules
import complexes
import reactions
import datareader
import evaluator
import ratekernel


print(f'{" set system parameters ":#^80}') #####################################

# maximum length of considered oligomer
n_max = 7
# length of template
Lt = 10
# maximum number of oligomers hybridized to template
degree_total = 4
# maximum number of O-Acylisourea oligomers hybridized to template
degree_O = 0
# maximum number of N-Acylisourea oligomers hybridized to template
degree_N = 1


print(f'{" initialize chemical compounds ":#^80}') #############################

# initialize table listing all chemical compounds on template
table_full = complexes.generate_table_full_complexity(n_max, Lt, \
    degree_total, degree_O, degree_N)

# initialize table listing all chemical compounds on template in reduced complexity
# space (summarizing compounds of equal hybridization energy into one class of compounds)
table_reduced = complexes.generate_table_reduced_complexity(table_full, Lt)

# determine map between full complexity and reduced complexity representation
map_reduced_full = complexes.generate_map_reduced_full(table_full, table_reduced, Lt)

# initialize table listing all chemical compounds in solution
table_sol = complexes.generate_table_full_complexity(n_max, 0, 0, 0, 0)

# initialize table listing all chemical compounds in solution in reduced complexity
# representation
table_sol_reduced = complexes.generate_table_reduced_complexity(table_sol, Lt)

# determine complex properties, e. g. which and how many oligomers belong to 
# each oligomer-template complex
affiliation_oligomer_reduced_complex_reduced = \
    complexes.list_affiliations_oligomers_reduced_complexity(table_sol_reduced, \
    table_reduced, map_reduced_full)
properties_complex_reduced = complexes.construct_complex_properties_reduced_complexity(\
    table_sol_reduced, affiliation_oligomer_reduced_complex_reduced)
properties_complex_full = complexes.construct_complex_properties_full_complexity(\
    table_sol, table_full, Lt)


print(f'{" initialize chemical reactions ":#^80}') #############################

# initialize table that lists all reaction rate constants for reactions in solution
table_rate_constants_sol = ratekernel.list_rate_constants()

# list all chemical reactions in solution
acts_sol, acts_sol_humanreadable = reactions.list_activations_solution(n_max, table_sol)
ligs_sol, ligs_sol_humanreadable = reactions.list_ligations_solution(n_max, table_sol)
losses_sol, losses_sol_humanreadable = reactions.list_losses_solution(n_max, table_sol)
hydros_sol, hydros_sol_humanreadable = reactions.list_hydrolysis_solution(n_max, table_sol)
cuts_sol, cuts_sol_humanreadable = reactions.list_cleavages_solution(n_max, table_sol)

# list all chemical reactions on template
ligs_temp, ligs_temp_humanreadable = reactions.list_ligations_template_simplified(\
    n_max, table_full)
losses_temp, losses_temp_humanreadable = reactions.list_losses_template_simplified(\
    table_full)
hydros_temp, hydros_temp_humanreadable = reactions.list_hydrolysis_template_simplified(\
    table_full)
cuts_temp2temp, cuts_temp2temp_humanreadable = reactions.list_cleavages_temp2temp(\
    Lt, degree_total, table_full)
cuts_temp2sol, cuts_temp2sol_humanreadable = reactions.list_cleavages_temp2sol(\
    Lt, table_full)


print(f'{" read rate constants and dissociation constants ":#^80}') ############

# choose parameter set of interest
# TODO: set name of parameter set
name = 0
print("parmeter set of interest: ", name)

# read reaction rate constants on template and dissociation constants
ks_temp_Kds_log = datareader.read_parameters_system_with_template(\
    './parameters.xlsx', name, Lt)

# read reaction rate constants in solution
# note: these reaction rate constants are identical to the first parameter set in
# ../without_template__length-dependent_rate_constants/rate_constants_variable_klig.xlsx
ks_sol = np.loadtxt('./rate_constants_sol.txt')

# combine all reaction rate constants and dissociation constants into one array
ks_log = np.concatenate((np.log2(ks_sol), ks_temp_Kds_log))


print(f'{" compute time evolution ":#^80}') ####################################

# read experimental data
times_exp, map_oligos_exp, c_oligos_exp, c_oligos_exp_err, c_EDC_exp, c_EDC_exp_err = \
    datareader.read_experimental_data_T25_EDC10_A10()

# initial concentration
c_full_initial = np.zeros(len(table_full)+1)
# useable monomer concentration
c_full_initial[table_full['1,_,0']] = 25.
# template concentration
c_full_initial[table_full['']] = 0.8
# EDC concentration
c_full_initial[-1] = 10.

# compute time evolution
times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, cs = \
    evaluator.model_solution_kernel_template_scalar(\
    ks_log, c_full_initial, times_exp, c_oligos_exp, map_oligos_exp, c_EDC_exp, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    ligs_temp, losses_temp, hydros_temp, cuts_temp2temp, cuts_temp2sol, \
    table_sol, table_sol_reduced, table_full, table_rate_constants_sol, n_max, Lt, \
    properties_complex_full, properties_complex_reduced, \
    affiliation_oligomer_reduced_complex_reduced, True, True, False)

# compute concentration of oligomers in solution
c_oligos_theo_sol = np.zeros(np.shape(c_oligos_theo))
for i in range(len(c_oligos_theo)):
    c_oligos_theo_sol[i] = np.asarray([cs[it][i+1] for it in range(len(cs))])

# plot time evolution
legend_entries = ['total', 'in solution']
times_theo_list = [times_theo, times_theo]
c_oligos_theo_list = [c_oligos_theo, c_oligos_theo_sol]
map_oligos_theo_list = [map_oligos_theo, map_oligos_theo]
c_EDC_theo_list = [c_EDC_theo, c_EDC_theo]

evaluator.plot_trajectories_template(times_exp, c_oligos_exp, map_oligos_exp, \
    c_oligos_exp_err, c_EDC_exp, c_EDC_exp_err, legend_entries, times_theo_list, \
    c_oligos_theo_list, map_oligos_theo_list, c_EDC_theo_list, True, \
    './time_evolution_%s.pdf' %name)

# save time evolution to pickle file
out = [times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, cs]
f = open('./timeevolution_%s.pkl' %name, 'wb')
pkl.dump(out, f)
f.close()