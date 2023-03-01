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


print(f'{" compute mean oligomer length ":#^80}') ##############################

# determine oligomer length for each oligomer
lengths = np.asarray([int(key.split(',')[0]) for key in map_oligos_theo.keys()])

# compute mean oligomer length
c_oligos_theo_sol_weighted = (lengths * c_oligos_theo_sol.T).T
c_oligos_theo_temp_weighted = (lengths * c_oligos_theo_temp.T).T

l_avg_sol = np.sum(c_oligos_theo_sol_weighted, axis=0)/np.sum(c_oligos_theo_sol, axis=0)
l_avg_temp = np.sum(c_oligos_theo_temp_weighted, axis=0)/np.sum(c_oligos_theo_temp, axis=0)

# plot mean oligomer length
f = plt.figure(figsize=(4.5,3), constrained_layout=True)
plt.plot(times_theo, l_avg_sol, label='in solution')
plt.plot(times_theo, l_avg_temp, label='on template')
plt.xlabel('time (h)')
plt.ylabel(r'average oligomer length $\langle l \rangle$')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('./average_oligomer_length_%s.pdf' %name)
plt.show()
