#!/env/bin python3.9

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, '../src')


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

# compute concentration on template
c_oligos_theo_temp = c_oligos_theo - c_oligos_theo_sol


print(f'{" compute percentages of oligomer concentration ":#^80}') #############

# pick compounds of interest
compound_names = ['DynT', 'DynT$^*$', 'DynT$_2$', 'DynT$_2^*$', 'DynT$_3$', \
    'DynT$_3^*$', 'DynT$_4$', 'DynT$_5$']
compound_keys = ['1,_,0', '1,_,2', '2,_,0', '2,_,2', '3,_,0', '3,_,2', '4,_,0', \
    '5,_,0']
indices = sorted(np.asarray([map_oligos_theo[oligo] for oligo in compound_keys], \
    dtype=int))

# compute percentages of concentrations
c_oligos_theo_sol_rel = c_oligos_theo_sol[indices]
c_oligos_theo_temp_rel = c_oligos_theo_temp[indices]

c_oligos_theo_sol_total = np.sum(c_oligos_theo_sol, axis=0)
c_oligos_theo_sol_rel_perc = c_oligos_theo_sol_rel/c_oligos_theo_sol_total
c_oligos_theo_sol_rel_perc_cum = np.cumsum(c_oligos_theo_sol_rel_perc, axis=0)

c_oligos_theo_temp_total = np.sum(c_oligos_theo_temp, axis=0)
c_oligos_theo_temp_rel_perc = c_oligos_theo_temp_rel/c_oligos_theo_temp_total
c_oligos_theo_temp_rel_perc_cum = np.cumsum(c_oligos_theo_temp_rel_perc, axis=0)

# plot percentages of concentrations
f, ax = plt.subplots(1, 2, figsize=(9,3.5), constrained_layout=True)

ax[0].set_title('in solution')
ax[0].fill_between(times_theo, np.zeros(len(times_theo)), c_oligos_theo_sol_rel_perc_cum[0], \
    alpha = 0.8, label = compound_names[0])
for i in range(1, len(indices)):
    ax[0].fill_between(times_theo, c_oligos_theo_sol_rel_perc_cum[i-1], \
        c_oligos_theo_sol_rel_perc_cum[i], alpha = 0.8, label = compound_names[i])
ax[0].legend()
ax[0].set_xlabel('time (h)')
ax[0].set_ylabel('percentage of total concentration in solution')
ax[0].grid(alpha=0.3)

ax[1].set_title('on template')
ax[1].fill_between(times_theo, np.zeros(len(times_theo)), c_oligos_theo_temp_rel_perc_cum[0], \
    alpha=0.8, label = compound_names[0])
for i in range(1, len(indices)):
    ax[1].fill_between(times_theo, c_oligos_theo_temp_rel_perc_cum[i-1], \
        c_oligos_theo_temp_rel_perc_cum[i], alpha=0.8, label = compound_names[i])
ax[1].legend()
ax[1].set_xlabel('time (h)')
ax[1].set_ylabel('percentage of total concentration on template')
ax[1].grid(alpha=0.3)

plt.savefig('./oligomer_percentages_solution_vs_template_%s.pdf' %name)
plt.show()
