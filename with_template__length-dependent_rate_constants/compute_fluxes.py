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
import integrator


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

# list chemical reactions of interest in solution (sorted by altered oligomer type)"
acts_sol_inc, acts_sol_dec = \
    reactions.generate_table_complexes_solution_activation_productive_destructive(\
    table_sol, n_max)
ligs_sol_inc, ligs_sol_dec = \
    reactions.generate_table_complexes_solution_ligation_productive_destructive(\
    table_sol, n_max)
losses_sol_inc, losses_sol_dec = \
    reactions.generate_table_complexes_solution_loss_productive_destructive(\
    table_sol, n_max)
hydros_sol_inc, hydros_sol_dec = \
    reactions.generate_table_complexes_solution_hydro_productive_destructive(\
    table_sol, n_max)
cuts_sol_inc, cuts_sol_dec = \
    reactions.generate_table_complexes_solution_cleavage_productive_destructive(\
    table_sol, n_max)

# list chemical reactions of interest on template (sorted by altered oligomer type)"
ligs_temp_inc, ligs_temp_dec = \
    reactions.generate_table_complexes_templated_ligation_productive_destructive_restricted(\
    table_sol, table_full)
losses_temp_inc, losses_temp_dec = \
    reactions.generate_table_complexes_templated_loss_productive_destructive(\
    table_sol, table_full)


print(f'{" read rate constants and dissociation constants ":#^80}') ############

# choose parameter set of interest
# TODO: set name of parameter set
name = 0
print("parmeter set of interest: ", name)

# read reaction rate constants on template and dissociation constants
ks_temp_Kds_log = datareader.read_parameters_system_with_template(\
    './parameters.xlsx', name, Lt)
ks_temp = 2**ks_temp_Kds_log[0:4]
klig_temp, kloss_temp, khydro_temp, kcut_temp = ks_temp

# read reaction rate constants in solution
# note: these reaction rate constants are identical to the first parameter set in
# ../without_template__length-dependent_rate_constants/rate_constants_variable_klig.xlsx
ks_sol = np.loadtxt('./rate_constants_sol.txt')
ks_sol_log = np.log2(ks_sol)

# compute rate constant kernels based on ks_sol_log
Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol = \
    ratekernel.choose_rate_constant_kernels(2**ks_sol_log, \
    table_sol, table_rate_constants_sol, n_max)

# combine all reaction rate constants and dissociation constants into one array
ks_log = np.concatenate((np.log2(ks_sol), ks_temp_Kds_log))


print(f'{" read pre-computed time-evolution ":#^80}') ##########################

try:
    # read time-evolution
    f = open('./timeevolution_%s.pkl' %name, 'rb')
    out = pkl.load(f)
    f.close()
    times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, cs = out
except:
    raise ValueError('./timeevolution_%s.pkl does not exist: need to run \
./compute_timeevolution.py before ./compute_fluxes.py' %name)

# compute concentration of oligomers in solution
c_oligos_theo_sol = np.zeros(np.shape(c_oligos_theo))
for i in range(len(c_oligos_theo)):
    c_oligos_theo_sol[i] = np.asarray([cs[it][i+1] for it in range(len(cs))])


print(f'{" compute fluxes (production and consumption separated) ":#^80}') #####

# list of reactions of interest
reactions_rel = ['activation solution', 'ligation solution', 'loss solution', \
    'O-Acylisourea hydrolysis solution', 'anhydride hydrolysis solution', \
    'ligation template', 'loss template']

# fluxes due to reactions of interest in solution
dc_dt_acts_sol_inc, dc_dt_acts_sol_dec = \
    integrator.changes_activations_solution_kernel_fluxes(cs, Kact_sol, \
    acts_sol_inc, acts_sol_dec)
dc_dt_ligs_sol_inc, dc_dt_ligs_sol_dec = \
    integrator.changes_ligations_solution_kernel_fluxes(cs, Klig_sol, \
    ligs_sol_inc, ligs_sol_dec)
dc_dt_losses_sol_inc, dc_dt_losses_sol_dec = \
    integrator.changes_losses_solution_kernel_fluxes(cs, Kloss_sol, \
    losses_sol_inc, losses_sol_dec)
dc_dt_hydros_sol_inc, dc_dt_hydros_sol_dec = \
    integrator.changes_hydrolysis_solution_kernel_fluxes(cs, Khydro_sol, \
    hydros_sol_inc, hydros_sol_dec)
dc_dt_cuts_sol_inc, dc_dt_cuts_sol_dec = \
    integrator.changes_cleavages_solution_kernel_fluxes(cs, Kcut_sol, \
    cuts_sol_inc, cuts_sol_dec)

# fluxes due to reactions of interest on template
dc_dt_ligs_temp_inc, dc_dt_ligs_temp_dec = \
    integrator.changes_ligations_template_simplified_fluxes(cs, klig_temp, \
    ligs_temp_inc, ligs_temp_dec)
dc_dt_losses_temp_inc, dc_dt_losses_temp_dec = \
    integrator.changes_losses_template_simplified_fluxes(cs, kloss_temp, \
    losses_temp_inc, losses_temp_dec)

# write fluxes to list-structure
dc_dt_s_incdec = \
    [[dc_dt_acts_sol_inc,dc_dt_acts_sol_dec], \
    [dc_dt_ligs_sol_inc,dc_dt_ligs_sol_dec], \
    [dc_dt_losses_sol_inc,dc_dt_losses_sol_dec], \
    [dc_dt_hydros_sol_inc, dc_dt_hydros_sol_dec], \
    [dc_dt_cuts_sol_inc, dc_dt_cuts_sol_dec], \
    [dc_dt_ligs_temp_inc, dc_dt_ligs_temp_dec], \
    [dc_dt_losses_temp_inc, dc_dt_losses_temp_dec]]

# save fluxes
f = open('./fluxes_%s.pkl' %name, 'wb')
pkl.dump([reactions_rel, dc_dt_s_incdec], f)
f.close()


print(f'{" compute fluxes (production and consumption summed) ":#^80}') ########

# array to store cumulative fluxes of production and consumption
dc_dt_oligos = np.zeros((len(reactions_rel)+1, len(dc_dt_s_incdec[0][0][1:-1]), \
    len(dc_dt_s_incdec[0][0][1:-1][0])))
dc_dt_EDC = np.zeros((len(reactions_rel)+1, len(dc_dt_s_incdec[0][0][-1])))

# compute cumulative fluxes
for ir in range(len(reactions_rel)):
    dc_dt_oligos[ir+1] = dc_dt_s_incdec[ir][0][1:-1] - dc_dt_s_incdec[ir][1][1:-1]
    dc_dt_EDC[ir+1] = dc_dt_s_incdec[ir][0][-1] - dc_dt_s_incdec[ir][1][-1]

# compute flux due to all reactions
# note: only valid if k_{hydrolysis O-Acylisourea on template} = 0
# and k_{hydrolysis anhydride on template} = 0
dc_dt_oligos[0] = np.sum(dc_dt_oligos[1:], axis=0)
dc_dt_EDC[0] = np.sum(dc_dt_EDC[1:], axis=0)

# plot fluxes
evaluator.plot_fluxes_template(times_theo, dc_dt_oligos, \
    map_oligos_theo, dc_dt_EDC, ['all reactions', *reactions_rel], True, \
    "./fluxes_%s.pdf" %name)

# compute fraction of O-Acylisourea that undergoes ligation
frac_lig = evaluator.compute_relative_weight_of_ligation_single_solution(\
    2**ks_sol_log, c_oligos_theo_sol, map_oligos_theo, table_rate_constants_sol, n_max)
np.savetxt('./fraction_ligation_upon_activation_%s.txt' %name, frac_lig)