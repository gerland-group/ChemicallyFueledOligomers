#!/bin/env python3

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# own modules
import complexes
import integrator
import ratekernel



def interpolate_theory_curves(times_exp, times_theo, c_oligos_theo, c_EDC_theo, \
    map_oligos_exp, map_oligos_theo):
    
    # interpolate the theoretical oligomer trajectories at experimental time values
    c_oligos_theo_discrete = np.zeros((len(map_oligos_exp), len(times_exp)))
    for i, cmplx in enumerate(list(map_oligos_exp.keys())):
        c_rel = c_oligos_theo[map_oligos_theo[cmplx]]
        f = interpolate.interp1d(times_theo, c_rel)
        c_oligos_theo_discrete[i] = f(times_exp)

    # interpolate the theoretical EDC trajectory at experimental time values
    f = interpolate.interp1d(times_theo, c_EDC_theo)
    c_EDC_theo_discrete = f(times_exp)

    return c_oligos_theo_discrete, c_EDC_theo_discrete



def compute_residual(c_oligos_theo, c_EDC_theo, c_oligos_exp, c_EDC_exp, \
    times_exp, times_theo, map_oligos_exp, map_oligos_theo):

    # compute discrete versions of the theory arrays
    c_oligos_theo_discrete, c_EDC_theo_discrete = interpolate_theory_curves(\
        times_exp, times_theo, c_oligos_theo, c_EDC_theo, map_oligos_exp, map_oligos_theo)

    c_rel_theo = np.array([*c_oligos_theo_discrete, c_EDC_theo_discrete])
    c_rel_exp = np.array([*c_oligos_exp, c_EDC_exp])

    # compute the residual
    residual = 0

    for i in range(len(c_rel_exp)):
        for j in range(len(c_rel_exp[i])):
            mean = np.nanmean(c_rel_exp[i])
            if(c_rel_exp[i][j] == 0):
                residual += ((c_rel_exp[i][j] - c_rel_theo[i][j])**2)/(mean**2)
            elif(np.isnan(c_rel_exp[i][j])):
                continue
            else:
                residual += ((c_rel_exp[i][j] - c_rel_theo[i][j])**2)/(mean**2)
    
    return residual



def model_solution_scalar(ks_log, c_full_initial, times_exp, c_oligos_exp, \
    map_oligos_exp, c_EDC_exp, acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    table_sol, n_max, TRAJOUT, PRINTRES):

    # unpack the parameters
    ks = 2**ks_log

    kact_sol, klig_sol, kloss_sol, khydro_sol, kcut_sol = ks

    # fix start and end time of integration
    t_init = 0
    t_final = times_exp[-1]

    # integrate trajectory
    times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo = \
        integrator.integrate_trajectory_solution_scalar(t_init, t_final, c_full_initial, \
            kact_sol, klig_sol, kloss_sol, khydro_sol, kcut_sol, acts_sol, ligs_sol, losses_sol, \
            hydros_sol, cuts_sol, table_sol, n_max)
    
    # compute the residual
    res = compute_residual(c_oligos_theo, c_EDC_theo, c_oligos_exp, c_EDC_exp, \
        times_exp, times_theo, map_oligos_exp, map_oligos_theo)

    if(PRINTRES):
        print("res due to fit: ", res)

    if(TRAJOUT):
        return times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, res
    
    else:
        return res



def model_solution_kernel(ks_log, c_full_initial, times_exp, \
    c_oligos_exp, map_oligos_exp, c_EDC_exp, acts_sol, ligs_sol, losses_sol, hydros_sol, \
    cuts_sol, table_sol, table_rate_constant, n_max, alpha, TRAJOUT, PRINTRES, RESINFO=False):

    ks = 2**ks_log

    # compute reaction rate constants
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol =  \
        ratekernel.choose_rate_constant_kernels(\
        ks, table_sol, table_rate_constant, n_max)

    # fix start and end time of integration
    t_init = 0
    t_final = times_exp[-1]

    # integrate trajectory
    times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo = \
        integrator.integrate_trajectory_solution_kernel(t_init, t_final, c_full_initial, \
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, acts_sol, ligs_sol, losses_sol, \
    hydros_sol, cuts_sol, table_sol, n_max)
    
    # compute the residual
    res = compute_residual(c_oligos_theo, c_EDC_theo, c_oligos_exp, c_EDC_exp, \
        times_exp, times_theo, map_oligos_exp, map_oligos_theo)

    if(PRINTRES):
        print("res due to fit: ", res)

    #res += 1e-6*(end-start)**2
    rate_spread_penalty = alpha*ratekernel.compute_spanned_orders_of_magnitude(\
        ks, n_max)
    res += rate_spread_penalty

    if(PRINTRES):
        print("res due to rate spread penalty: ", rate_spread_penalty)
    
    if(TRAJOUT and RESINFO):
        return times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, \
            res, res-rate_spread_penalty, rate_spread_penalty
    
    elif(TRAJOUT and not RESINFO):
        return times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, res
    
    elif(not TRAJOUT and RESINFO):
        return res, res-rate_spread_penalty, rate_spread_penalty        

    else:
        return res



def model_solution_kernel_template_scalar(ks_log, c_full_initial, \
    times_exp, c_oligos_exp, map_oligos_exp, c_EDC_exp, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    ligs_temp, losses_temp, hydros_temp, cuts_temp2temp, cuts_temp2sol, \
    table_sol, table_sol_reduced, table_full, table_rate_constants_sol, n_max, Lt, \
    properties_complex_full, properties_complex_reduced, \
    affiliation_oligomer_reduced_complex_reduced, TRAJOUT, PRINTRES, COMPRES):

    ks = 2**ks_log
    ks_sol = ks[0:len(table_rate_constants_sol)]
    ks_temp = ks[len(table_rate_constants_sol):len(table_rate_constants_sol)+4]
    Kds = ks[len(table_rate_constants_sol)+4:]

    Kds_complex_reduced = complexes.construct_complex_Kds_reduced_complexity(\
        Kds, table_sol_reduced, affiliation_oligomer_reduced_complex_reduced)
    Kds_complex_full = complexes.construct_complex_Kds_full_complexity(\
        Kds, table_sol, table_full, Lt)

    # compute reaction rate constants
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol = \
        ratekernel.choose_rate_constant_kernels(\
        ks_sol, table_sol, table_rate_constants_sol, n_max)
    
    klig_temp, kloss_temp, khydro_temp, kcut_temp = ks_temp

    # fix start and end time of integration
    t_init = 0
    t_final = times_exp[-1]
    #t_final = 0.005

    # integrate trajectory
    times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, cs = \
        integrator.integrate_trajectory_solution_kernel_template_scalar(\
        t_init, t_final, c_full_initial, Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, \
        klig_temp, kloss_temp, khydro_temp, kcut_temp, Kds_complex_full, Kds_complex_reduced, \
        acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
        ligs_temp, losses_temp, hydros_temp, cuts_temp2temp, cuts_temp2sol, \
        properties_complex_full, properties_complex_reduced, n_max)
    
    # compute the residual
    if(COMPRES and PRINTRES):
        res = compute_residual(c_oligos_theo, c_EDC_theo, c_oligos_exp, c_EDC_exp, \
            times_exp, times_theo, map_oligos_exp, map_oligos_theo)
        print("res due to fit: ", res)

    elif(COMPRES and not PRINTRES):
        res = compute_residual(c_oligos_theo, c_EDC_theo, c_oligos_exp, c_EDC_exp, \
            times_exp, times_theo, map_oligos_exp, map_oligos_theo)
    
    if(COMPRES and TRAJOUT):
        return times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, res
    elif(not COMPRES and TRAJOUT):
        return times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, cs
    elif(COMPRES and not TRAJOUT):
        return res



def plot_trajectories_solution(times_exp, c_oligos_exp, map_oligos_exp, c_oligos_exp_err, \
    c_EDC_exp, c_EDC_exp_err, times_theo, c_oligos_theo, map_oligos_theo, c_EDC_theo, \
    SAVE, filename):

    keys = ['2,_,0', '3,_,0', '4,_,0', '5,_,0', '1,_,2']
    names = ['dimer', 'trimer', 'tetramer', 'pentamer', 'lost monomer']

    f = plt.figure(figsize=(9,9), constrained_layout=True)
    gs = f.add_gridspec(3, 4)
    grids = [gs[0,0:2], gs[0,2:], gs[1,0:2], gs[1,2:], gs[2,0:2], gs[2,2:]]

    for i in range(len(keys)):

        ax = f.add_subplot(grids[i])
        ax.set_title(names[i])
        ax.errorbar(times_exp, c_oligos_exp[map_oligos_exp[keys[i]]], \
            yerr=c_oligos_exp_err[map_oligos_exp[keys[i]]], linestyle='', \
            marker = 'x', label = 'with template')
        ax.plot(times_theo, c_oligos_theo[map_oligos_theo[keys[i]]], label = 'theory')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('concentration (mM)')
        ax.legend()
        ax.grid(alpha=0.3)

    ax = f.add_subplot(grids[-1])
    ax.set_title("EDC")
    ax.errorbar(times_exp, c_EDC_exp, yerr=c_EDC_exp_err, \
        linestyle='', marker='x', label='with template')
    ax.plot(times_theo, c_EDC_theo, label = 'theory')
    ax.set_xlabel('time (h)')
    ax.set_ylabel('concentration (mM)')
    ax.legend()
    ax.grid(alpha=0.3)

    if(SAVE):
        plt.savefig(filename)

    plt.show()



def plot_trajectories_template(times_exp, c_oligos_exp, map_oligos_exp, c_oligos_exp_err, \
    c_EDC_exp, c_EDC_exp_err, legend_entries, times_theo_list, c_oligos_theo_list, \
    map_oligos_theo_list, c_EDC_theo_list, SAVE, filename):
    
    keys = ['1,_,0', '2,_,0', '3,_,0', '4,_,0', '5,_,0', '1,_,2', '3,_,2']
    names = ['monomer', 'dimer', 'trimer', 'tetramer', 'pentamer', 'lost monomer', 'lost trimer']

    f = plt.figure(figsize=(9,12), constrained_layout=True)
    gs = f.add_gridspec(4, 4)
    grids = [gs[0,0:2], gs[0,2:], gs[1,0:2], gs[1,2:], gs[2,0:2], gs[2,2:], gs[3,0:2], gs[3,2:]]

    for i in range(len(keys)):

        ax = f.add_subplot(grids[i])
        ax.set_title(names[i])
        try:
            ax.errorbar(times_exp, c_oligos_exp[map_oligos_exp[keys[i]]], \
                yerr=c_oligos_exp_err[map_oligos_exp[keys[i]]], linestyle='', marker = 'x', \
                label = 'experiment')
        except:
            pass
            
        for j in range(len(times_theo_list)):
            ax.plot(times_theo_list[j], c_oligos_theo_list[j][map_oligos_theo_list[j][keys[i]]], \
                label = legend_entries[j])
        
        ax.set_xlabel('time (h)')
        ax.set_ylabel('concentration (mM)')
        ax.legend()
        ax.grid(alpha=0.3)

    ax = f.add_subplot(grids[-1])
    ax.set_title("EDC")
    ax.errorbar(times_exp, c_EDC_exp, yerr=c_EDC_exp_err, \
        linestyle='', marker='x', label='experiment')
    
    for j in range(len(times_theo_list)):
        ax.plot(times_theo_list[j], c_EDC_theo_list[j], label = legend_entries[j])
    
    ax.set_xlabel('time (h)')
    ax.set_ylabel('concentration (mM)')
    ax.legend()
    ax.grid(alpha=0.3)

    if(SAVE):
        plt.savefig(filename)

    plt.show()



def plot_fluxes_template(times_theo, dc_dt_oligos_theo, map_oligos_theo, \
    dc_dt_EDC_theo, reaction_types, SAVE, filename):

    keys = ['1,_,0', '2,_,0', '3,_,0', '4,_,0', '5,_,0']
    names = ['monomer', 'dimer', 'trimer', 'tetramer', 'pentamer']

    f = plt.figure(figsize=(9,9), constrained_layout=True)
    gs = f.add_gridspec(3,4)
    grids = [gs[0,0:2], gs[0,2:], gs[1,0:2], gs[1,2:], gs[2,0:2], gs[2,2:]]

    for i in range(len(keys)):

        ax = f.add_subplot(grids[i])
        ax.set_title(names[i])

        for j in range(len(reaction_types)):
            ax.plot(times_theo, dc_dt_oligos_theo[j][map_oligos_theo[keys[i]]], \
            label = reaction_types[j])
        
        ax.set_xlabel('time (h)')
        ax.set_ylabel('flux (= rate) (mM/h)')
        ax.legend()
        ax.grid(alpha=0.3)

    ax = f.add_subplot(grids[-1])
    ax.set_title("EDC")

    for j in range(len(reaction_types)):
        ax.plot(times_theo, dc_dt_EDC_theo[j], label = reaction_types[j])
    
    ax.set_xlabel('time (h)')
    ax.set_ylabel('flux (= rate) (mM/h)')
    ax.legend()
    ax.grid(alpha=0.3)

    if(SAVE):
        plt.savefig(filename)

    plt.show()



def compute_relative_weight_of_hydrolysis_solution(ks, c_oligos_theo, \
    map_oligos_theo, table_rate_constant, n_max):

    numerator = ks[table_rate_constant['hydro']]

    denominator = ks[table_rate_constant['hydro']] + ks[table_rate_constant['loss']]

    for m in range(1, n_max+1):
        denominator += ks[table_rate_constant['lig']]*c_oligos_theo[map_oligos_theo['%s,_,0' %m]]

    gamma = numerator/denominator

    return gamma



def compute_relative_weight_of_ligation_single_solution(ks, c_oligos_theo, \
    map_oligos_theo, table_rate_constant, n_max):

    numerator = ks[table_rate_constant['hydro']] + ks[table_rate_constant['loss']]

    denominator = ks[table_rate_constant['hydro']] + ks[table_rate_constant['loss']]

    for m in range(1, n_max+1):
        denominator += ks[table_rate_constant['lig']]*c_oligos_theo[map_oligos_theo['%s,_,0' %m]]
    
    return 1 - numerator/denominator



def compute_EDC_consumption_sol_temp_ligloss(times_theo, dc_dt_s_incdec, \
    reactions_rel, table_sol):

    # list compounds for which the EDC consumption is supposed to be computed
    compounds_rel = ['1,_,0', '2,_,0', '3,_,0', '4,_,0', '5,_,0']
    indices_compounds_rel = sorted(np.asarray([table_sol[compound] for compound \
        in compounds_rel], dtype=int))

    # compute EDC-consumption in solution
    EDC_consumed_sol_ligloss = np.zeros(len(table_sol))

    for ic in range(len(table_sol)):
        # consumption of EDC via activation of oligomers
        EDC_consumed_sol_ligloss[ic] = np.trapz(\
            dc_dt_s_incdec[reactions_rel.index('activation solution')][1][ic], \
            x = times_theo)

    EDC_consumed_sol_ligloss = EDC_consumed_sol_ligloss[indices_compounds_rel]

    # compute EDC-consumption on template
    EDC_consumed_temp_ligloss = np.zeros(len(table_sol))

    for ic in range(len(table_sol)):
        # consumption of EDC via ligation of two oligomers
        EDC_consumed_temp_ligloss[ic] = np.trapz(\
            1/2 * dc_dt_s_incdec[reactions_rel.index('ligation template')][1][ic], \
            x = times_theo)
        
        # consumption of EDC via side product formation
        EDC_consumed_temp_ligloss[ic] += np.trapz(\
            dc_dt_s_incdec[reactions_rel.index('loss template')][1][ic], \
            x = times_theo)

    EDC_consumed_temp_ligloss = EDC_consumed_temp_ligloss[indices_compounds_rel]

    return EDC_consumed_sol_ligloss, EDC_consumed_temp_ligloss



def compute_EDC_consumption_sol_temp_lig(times_theo, dc_dt_s_incdec, \
    reactions_rel, table_sol, frac_lig):
    
    # list compounds for which the EDC consumption is supposed to be computed  
    compounds_rel = {'1,_,0', '2,_,0', '3,_,0', '4,_,0', '5,_,0'}
    indices_compounds_rel = sorted(np.asarray([table_sol[compound] for compound \
        in compounds_rel], dtype=int))

    # compute EDC-consumption in solution
    EDC_consumed_sol_lig = np.zeros(len(table_sol))

    for ic in range(len(table_sol)):
        # consumption of EDC via activation of oligomers
        EDC_consumed_sol_lig[ic] = np.trapz(\
            frac_lig*dc_dt_s_incdec[reactions_rel.index('activation solution')][1][ic], \
            x = times_theo)

    EDC_consumed_sol_lig = EDC_consumed_sol_lig[indices_compounds_rel]

    # compute EDC-consumption on template
    EDC_consumed_temp_lig = np.zeros(len(table_sol))

    for ic in range(len(table_sol)):
        # consumption of EDC via ligation of two oligomers
        EDC_consumed_temp_lig[ic] = np.trapz(\
            1/2 * dc_dt_s_incdec[reactions_rel.index('ligation template')][1][ic], \
            x = times_theo)

    EDC_consumed_temp_lig = EDC_consumed_temp_lig[indices_compounds_rel]

    return EDC_consumed_sol_lig, EDC_consumed_temp_lig
