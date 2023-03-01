#!/env/bin python3.9

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, '../src')

# import own modules
import complexes
import evaluator


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

# initialize table listing all chemical compounds in solution
table_sol = complexes.generate_table_full_complexity(n_max, 0, 0, 0, 0)


print(f'{" read rate constants and dissociation constants ":#^80}') ############

# choose parameter set of interest
# TODO: set name of parameter set
name = 0
print("parmeter set of interest: ", name)


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


print(f'{" read pre-computed fluxes ":#^80}') ##################################

try:
    # read fluxes
    f = open('./fluxes_%s.pkl' %name, 'rb')
    out = pkl.load(f)
    f.close()
    reactions_rel, dc_dt_s_incdec = out

    # read fraction of ligation upon activation
    frac_lig = np.loadtxt('./fraction_ligation_upon_activation_%s.txt' %name)

except:
    raise ValueError('./fluxes_%s.pkl does not exist: need to run \
./compute_fluxes.py before ./compute_EDC_consumption.py' %name)


print(f'{" compute EDC-consuption due to ligation and side product formation ":#^80}')

# compute EDC consumption via ligation and loss
EDC_consumed_sol_ligloss, EDC_consumed_temp_ligloss = \
    evaluator.compute_EDC_consumption_sol_temp_ligloss(times_theo, dc_dt_s_incdec, \
    reactions_rel, table_sol)

# plot EDC consumption
f, ax = plt.subplots(1, 2, figsize=(9,3), constrained_layout=True)

ax[0].set_title('in solution')
ax[0].bar(np.arange(len(EDC_consumed_sol_ligloss)), EDC_consumed_sol_ligloss)
ax[0].set_xticks(np.arange(len(EDC_consumed_sol_ligloss)), ['DynT', 'DynT$_2$', \
    'DynT$_3$', 'DynT$_4$', 'DynT$_5$'])
ax[0].set_ylabel('concentration of consumed EDC (mM)')
ax[0].grid(alpha=0.3)

ax[1].set_title('on template')
ax[1].bar(np.arange(len(EDC_consumed_temp_ligloss)), EDC_consumed_temp_ligloss)
ax[1].set_xticks(np.arange(len(EDC_consumed_temp_ligloss)), ['DynT', 'DynT$_2$', \
    'DynT$_3$', 'DynT$_4$', 'DynT$_5$'])
ax[1].set_ylabel('concentration of consumed EDC (mM)')
ax[1].grid(alpha=0.3)

plt.savefig('./consumed_EDC_ligloss_%s.pdf' %name)

plt.show()


print(f'{" compute EDC-consuption due to ligation ":#^80}') ####################

# compute EDC consumption via ligation
EDC_consumed_sol_lig, EDC_consumed_temp_lig = \
    evaluator.compute_EDC_consumption_sol_temp_lig(times_theo, dc_dt_s_incdec, \
    reactions_rel, table_sol, frac_lig)

# plot EDC consumption
f, ax = plt.subplots(1, 2, figsize=(9,3), constrained_layout=True)

ax[0].set_title('in solution')
ax[0].bar(np.arange(len(EDC_consumed_sol_lig)), EDC_consumed_sol_lig)
ax[0].set_xticks(np.arange(len(EDC_consumed_sol_lig)), ['DynT', 'DynT$_2$', \
    'DynT$_3$', 'DynT$_4$', 'DynT$_5$'])
ax[0].set_ylabel('concentration of consumed EDC (mM)')
ax[0].grid(alpha=0.3)

ax[1].set_title('on template')
ax[1].bar(np.arange(len(EDC_consumed_temp_lig)), EDC_consumed_temp_lig)
ax[1].set_xticks(np.arange(len(EDC_consumed_temp_lig)), ['DynT', 'DynT$_2$', \
    'DynT$_3$', 'DynT$_4$', 'DynT$_5$'])
ax[1].set_ylabel('concentration of consumed EDC (mM)')
ax[1].grid(alpha=0.3)

plt.savefig('./consumed_EDC_lig_%s.pdf' %name)

plt.show()
