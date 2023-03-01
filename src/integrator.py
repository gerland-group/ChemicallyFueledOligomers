#!/bin/env python3

import numpy as np
from numba import njit
from scipy import integrate
from functools import partial
from scipy import optimize

# import own modules
import complexes



@njit
def change_activations_solution(c_full, kact_sol, acts_sol):

    dc_dt = np.zeros(len(c_full))

    ins = acts_sol[:,0:2]
    outs = acts_sol[:,2]

    interim = kact_sol*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]
    
    return dc_dt



@njit
def change_activations_solution_kernel(c_full, Kact_sol, acts_sol):

    dc_dt = np.zeros(len(c_full))

    ins = acts_sol[:,0:2]
    outs = acts_sol[:,2]

    Karray = Kact_sol[ins[:,0]]

    interim = Karray*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]
    
    return dc_dt



@njit
def change_activations_solution_kernel_insonly(c_full, Kact_sol, acts_sol_ins):

    Karray = Kact_sol[acts_sol_ins[:,0]]

    dc_dt = np.sum(Karray*c_full[acts_sol_ins[:,0]]*c_full[acts_sol_ins[:,1]])

    return dc_dt



def changes_activations_solution_kernel_fluxes(cs, Kact_sol, acts_sol_inc, acts_sol_dec):

    dc_dt_inc = np.zeros((len(acts_sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(acts_sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in acts_sol_inc.keys():

            if(len(acts_sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_activations_solution_kernel_insonly(cs[it], \
                    Kact_sol, acts_sol_inc[io])
            
            if(len(acts_sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_activations_solution_kernel_insonly(cs[it], \
                    Kact_sol, acts_sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_ligations_solution(c_full, klig_sol, ligs_sol):

    dc_dt = np.zeros(len(c_full))

    ins = ligs_sol[:,0:2]
    outs = ligs_sol[:,2]

    interim = klig_sol*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_ligations_solution_kernel(c_full, Klig_sol, ligs_sol):

    dc_dt = np.zeros(len(c_full))

    ins = ligs_sol[:,0:2]
    outs = ligs_sol[:,2]

    Karray = np.zeros(len(ins))

    for i in range(len(ins)):
        Karray[i] = Klig_sol[ins[i][0], ins[i][1]]

    interim = Karray*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_ligations_solution_kernel_insonly(c_full, Klig_sol, ligs_sol_ins):

    Karray = np.zeros(len(ligs_sol_ins))

    for i in range(len(ligs_sol_ins)):
        Karray[i] = Klig_sol[ligs_sol_ins[i][0], ligs_sol_ins[i][1]]
    
    dc_dt = np.sum(Karray * c_full[ligs_sol_ins[:,0]] * c_full[ligs_sol_ins[:,1]])

    return dc_dt



def changes_ligations_solution_kernel_fluxes(cs, Klig_sol, ligs_sol_inc, ligs_sol_dec):

    dc_dt_inc = np.zeros((len(ligs_sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(ligs_sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in ligs_sol_inc.keys():

            if(len(ligs_sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_ligations_solution_kernel_insonly(cs[it], \
                    Klig_sol, ligs_sol_inc[io])
            
            if(len(ligs_sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_ligations_solution_kernel_insonly(cs[it], \
                    Klig_sol, ligs_sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_losses_solution(c_full, kloss_sol, losses_sol):

    dc_dt = np.zeros(len(c_full))

    ins = losses_sol[:,0]
    outs = losses_sol[:,1]

    interim = kloss_sol*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_losses_solution_kernel(c_full, Kloss_sol, losses_sol):
    
    dc_dt = np.zeros(len(c_full))

    ins = losses_sol[:,0]
    outs = losses_sol[:,1]

    Karray = Kloss_sol[ins]

    interim = Karray*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_losses_solution_kernel_insonly(c_full, Kloss_sol, losses_sol_ins):

    Karray = Kloss_sol[losses_sol_ins]

    dc_dt = np.sum(Karray*c_full[losses_sol_ins])
    
    return dc_dt



def changes_losses_solution_kernel_fluxes(cs, Kloss_sol, losses_sol_inc, losses_sol_dec):

    dc_dt_inc = np.zeros((len(losses_sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(losses_sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in losses_sol_inc.keys():

            if(len(losses_sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_losses_solution_kernel_insonly(cs[it], \
                    Kloss_sol, losses_sol_inc[io])
                
            if(len(losses_sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_losses_solution_kernel_insonly(cs[it], \
                    Kloss_sol, losses_sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_hydrolysis_solution(c_full, khydro_sol, hydros_sol):

    dc_dt = np.zeros(len(c_full))

    ins = hydros_sol[:,0]
    outs = hydros_sol[:,1]

    interim = khydro_sol*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_hydrolysis_solution_kernel(c_full, Khydro_sol, hydros_sol):

    dc_dt = np.zeros(len(c_full))

    ins = hydros_sol[:,0]
    outs = hydros_sol[:,1]

    Karray = Khydro_sol[ins]

    interim = Karray*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_hydrolysis_solution_kernel_insonly(c_full, Khydro_sol, hydros_sol_ins):

    Karray = Khydro_sol[hydros_sol_ins]

    dc_dt = np.sum(Karray*c_full[hydros_sol_ins])

    return dc_dt



def changes_hydrolysis_solution_kernel_fluxes(cs, Khydro_sol, hydros_sol_inc, hydros_sol_dec):

    dc_dt_inc = np.zeros((len(hydros_sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(hydros_sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in hydros_sol_inc.keys():

            if(len(hydros_sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_hydrolysis_solution_kernel_insonly(cs[it], \
                    Khydro_sol, hydros_sol_inc[io])
                
            if(len(hydros_sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_hydrolysis_solution_kernel_insonly(cs[it], \
                    Khydro_sol, hydros_sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_cleavages_solution(c_full, kcut_sol, cuts_sol):

    dc_dt = np.zeros(len(c_full))

    ins = cuts_sol[:,0]
    outs = cuts_sol[:,1:]

    interim = kcut_sol*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i,0]] += interim[i]
        dc_dt[outs[i,1]] += interim[i]
    
    return dc_dt



@njit
def change_cleavages_solution_kernel(c_full, Kcut_sol, cuts_sol):

    dc_dt = np.zeros(len(c_full))

    ins = cuts_sol[:,0]
    outs = cuts_sol[:,1:]

    Karray = Kcut_sol[ins]

    interim = Karray*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i,0]] += interim[i]
        dc_dt[outs[i,1]] += interim[i]
    
    return dc_dt



@njit
def change_cleavages_solution_kernel_insonly(c_full, Kcut_sol, cuts_sol_ins):

    Karray = Kcut_sol[cuts_sol_ins]

    dc_dt = np.sum(Karray*c_full[cuts_sol_ins])

    return dc_dt



def changes_cleavages_solution_kernel_fluxes(cs, Kcut_sol, cuts_sol_inc, cuts_sol_dec):

    dc_dt_inc = np.zeros((len(cuts_sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(cuts_sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in cuts_sol_inc.keys():

            if(len(cuts_sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_cleavages_solution_kernel_insonly(cs[it], \
                    Kcut_sol, cuts_sol_inc[io])

            if(len(cuts_sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_cleavages_solution_kernel_insonly(cs[it], \
                    Kcut_sol, cuts_sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_activation_template(c_full, kact_temp, acts_temp):

    dc_dt = np.zeros(len(c_full))

    ins = acts_temp[:,0:2]
    outs = acts_temp[:,2]

    interim = kact_temp*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_ligations_template(c_full, klig_temp, ligs_temp):

    dc_dt = np.zeros(len(c_full))

    ws = ligs_temp[:,0]
    ins = ligs_temp[:,1].astype('int')
    outs = ligs_temp[:,2].astype('int')

    interim = klig_temp*ws*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_ligations_template_simplified(c_full, klig_temp, ligs_temp):

    dc_dt = np.zeros(len(c_full))

    ins = ligs_temp[:,0:2]
    outs = ligs_temp[:,2]

    interim = klig_temp*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i]] += interim[i]
    
    return dc_dt



@njit
def change_ligations_template_simplified_insonly(c_full, klig_temp, ligs_temp_ins):

    dc_dt = np.sum(klig_temp*c_full[ligs_temp_ins[:,0]]*c_full[ligs_temp_ins[:,1]])

    return dc_dt



def changes_ligations_template_simplified_fluxes(cs, klig_temp, ligs_temp_inc, ligs_temp_dec):

    dc_dt_inc = np.zeros((len(ligs_temp_inc), len(cs)))
    dc_dt_dec = np.zeros((len(ligs_temp_dec), len(cs)))

    for it in range(len(cs)):

        for io in ligs_temp_inc.keys():

            if(len(ligs_temp_inc[io]) != 0):
                dc_dt_inc[io,it] = change_ligations_template_simplified_insonly(cs[it], \
                    klig_temp, ligs_temp_inc[io])
            
            if(len(ligs_temp_dec[io]) != 0):
                dc_dt_dec[io,it] = change_ligations_template_simplified_insonly(cs[it], \
                    klig_temp, ligs_temp_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_losses_template(c_full, kloss_temp, losses_temp):

    dc_dt = np.zeros(len(c_full))

    ins = losses_temp[:,0]
    outs = losses_temp[:,1]

    interim = kloss_temp*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_losses_template_simplified(c_full, kloss_temp, losses_temp):

    dc_dt = np.zeros(len(c_full))

    ins = losses_temp[:,0:2]
    outs = losses_temp[:,2:]

    interim = kloss_temp*c_full[ins[:,0]]*c_full[ins[:,1]]

    for i in range(len(ins)):
        dc_dt[ins[i,0]] -= interim[i]
        dc_dt[ins[i,1]] -= interim[i]
        dc_dt[outs[i,0]] += interim[i]
        dc_dt[outs[i,1]] += interim[i]
    
    return dc_dt



@njit
def change_losses_template_simplified_insonly(c_full, kloss_temp, losses_temp_ins):

    dc_dt = np.sum(kloss_temp*c_full[losses_temp_ins[:,0]]*c_full[losses_temp_ins[:,1]])

    return dc_dt



def changes_losses_template_simplified_fluxes(cs, kloss_temp, losses_temp_inc, losses_temp_dec):

    dc_dt_inc = np.zeros((len(losses_temp_inc), len(cs)))
    dc_dt_dec = np.zeros((len(losses_temp_dec), len(cs)))

    for it in range(len(cs)):

        for io in losses_temp_inc.keys():

            if(len(losses_temp_inc[io]) != 0):
                dc_dt_inc[io,it] = change_losses_template_simplified_insonly(cs[it], \
                    kloss_temp, losses_temp_inc[io])
            
            if(len(losses_temp_dec[io]) != 0):
                dc_dt_dec[io,it] = change_losses_template_simplified_insonly(cs[it], \
                    kloss_temp, losses_temp_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_hydrolysis_template(c_full, khydro_temp, hydros_temp):

    dc_dt = np.zeros(len(c_full))

    ins = hydros_temp[:,0]
    outs = hydros_temp[:,1]

    interim = khydro_temp*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]

    return dc_dt



@njit
def change_hydrolysis_template_simplified(c_full, khydro_temp, hydros_temp):

    dc_dt = np.zeros(len(c_full))

    # using a simplified computation as only interested in the change in EDC;
    # oligomer concentration do not need to be updated!

    oligo_indices = hydros_temp[:,0]

    # compute change in EDC concentration
    dc_dt[-1] -= khydro_temp*c_full[-1]*np.sum(c_full[oligo_indices])

    return dc_dt



@njit
def change_cleavage_temp2temp(c_full, kcut_temp, cuts_temp2temp):

    dc_dt = np.zeros(len(c_full))

    ws = cuts_temp2temp[:,0]
    #ins = np.array([*cuts_temp2temp[:,1]], dtype=int)
    ins = cuts_temp2temp[:,1].astype('int')
    #outs = np.array([*cuts_temp2temp[:,2]], dtype=int)
    outs = cuts_temp2temp[:,2].astype('int')

    interim = kcut_temp*ws*c_full[ins]

    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i]] += interim[i]
    
    return dc_dt



@njit
def change_cleavage_temp2temp_insonly(c_full, kcut_temp, cuts_temp2temp_ins):

    ws = cuts_temp2temp_ins[:,0]
    ins = cuts_temp2temp_ins[:,1].astype('int')

    dc_dt = np.sum(kcut_temp*ws*c_full[ins])

    return dc_dt



def changes_cleavage_temp2temp_fluxes(cs, kcut_temp, cuts_temp2temp_inc, cuts_temp2temp_dec):

    dc_dt_inc = np.zeros((len(cuts_temp2temp_inc), len(cs)))
    dc_dt_dec = np.zeros((len(cuts_temp2temp_dec), len(cs)))

    for it in range(len(cs)):

        for io in cuts_temp2temp_inc.keys():

            if(len(cuts_temp2temp_inc[io]) != 0):
                dc_dt_inc[io,it] = change_cleavage_temp2temp_insonly(cs[it], \
                    kcut_temp, cuts_temp2temp_inc[io])
            
            if(len(cuts_temp2temp_dec[io]) != 0):
                dc_dt_dec[io,it] = change_cleavage_temp2temp_insonly(cs[it], \
                    kcut_temp, cuts_temp2temp_dec[io])
    
    return dc_dt_inc, dc_dt_dec



@njit
def change_cleavage_temp2sol(c_full, kcut_temp, cuts_temp2sol):

    dc_dt = np.zeros(len(c_full))

    ws = cuts_temp2sol[:,0]
    #ins = np.array([*cuts_temp2sol[:,1]], dtype=int)
    ins = cuts_temp2sol[:,1].astype('int')
    #outs = np.zeros((len(cuts_temp2sol), 2), dtype=int)
    #for i in range(len(cuts_temp2sol)):
    #    outs[i,0] = int(cuts_temp2sol[i,2])
    #    outs[i,1] = int(cuts_temp2sol[i,3])
    outs = cuts_temp2sol[:,2:].astype('int')
    
    interim = kcut_temp*ws*c_full[ins]
    
    for i in range(len(ins)):
        dc_dt[ins[i]] -= interim[i]
        dc_dt[outs[i,0]] += interim[i]
        dc_dt[outs[i,1]] += interim[i]
    
    return dc_dt



@njit
def change_cleavage_temp2sol_insonly(c_full, kcut_temp, cuts_temp2sol_ins):

    ws = cuts_temp2sol_ins[:,0]
    ins = cuts_temp2sol_ins[:,1].astype('int')

    dc_dt = np.sum(kcut_temp*ws*c_full[ins])

    return dc_dt



def changes_cleavage_temp2sol_fluxes(cs, kcut_temp, cuts_temp2sol_inc, cuts_temp2sol_dec):

    dc_dt_inc = np.zeros((len(cuts_temp2sol_inc), len(cs)))
    dc_dt_dec = np.zeros((len(cuts_temp2sol_dec), len(cs)))

    for it in range(len(cs)):

        for io in cuts_temp2sol_inc.keys():

            if(len(cuts_temp2sol_inc[io]) != 0):
                dc_dt_inc[io,it] = change_cleavage_temp2sol_insonly(cs[it], \
                    kcut_temp, cuts_temp2sol_inc[io])
            
            if(len(cuts_temp2sol_dec[io]) != 0):
                dc_dt_dec[io,it] = change_cleavage_temp2sol_insonly(cs[it], \
                    kcut_temp, cuts_temp2sol_dec[io])
    
    return dc_dt_inc, dc_dt_dec


################ COMPUTATION OF THE HYBRIDIZATION EQUILIBRIUM ##################


@njit
def compute_total_oligomer_concentration(c_full, ps):

    c_full_subs = c_full[0:-1]

    return np.sum(ps*c_full_subs, axis=1)



def mass_conservation(c_oligos, c_oligos_total, Kds_complex_reduced, properties_complex_reduced):
    
    diff = np.zeros(len(properties_complex_reduced))

    for i in range(len(properties_complex_reduced)):

        properties1 = properties_complex_reduced[i][:,0:3]
        properties2 = properties_complex_reduced[i][:,3:].astype('int')
        Kdprods = Kds_complex_reduced[i]

        diff[i] = np.sum(properties1[:,0] * 1/Kdprods * np.prod(c_oligos**properties2, axis=1)) - c_oligos_total[i]
    
    return diff



def compute_complex_concentration(c_equ_sol, c_EDC, Kds_complex_full, ps, n_sol, n_full):

    c_equ_full = np.zeros(n_full+1)

    c_equ_sol_inter = np.zeros((n_sol, n_full))
    for i in range(n_full):
        c_equ_sol_inter[:,i] = c_equ_sol
    
    c_equ_full[0:-1] = 1/Kds_complex_full * np.prod((c_equ_sol_inter)**ps, axis=0)

    # set EDC concentration
    c_equ_full[-1] = c_EDC
    
    return c_equ_full



def compute_equilibrium_concentration(c_full, Kds_complex_full, Kds_complex_reduced, \
    properties_complex_full, properties_complex_reduced):

    ds = properties_complex_full[0]
    ls = properties_complex_full[1]
    ps = properties_complex_full[2:]

    n_sol = len(properties_complex_reduced)
    n_full = len(properties_complex_full[0])

    # save EDC concentration
    c_EDC = c_full[-1]
    
    # compute total oligomer concentration
    c_oligos_total = compute_total_oligomer_concentration(c_full, ps)

    # find equilibrium concentration
    
    c_equ_sol = optimize.root(mass_conservation, c_full[0:n_sol], args=(c_oligos_total, \
        Kds_complex_reduced, properties_complex_reduced), options={'xtol':1.49012e-30}).x
    
    c_equ_full = compute_complex_concentration(c_equ_sol, c_EDC, Kds_complex_full, \
        ps, n_sol, n_full)

    # c_equ_full = np.round(c_equ_full, 12)

    return c_equ_full


######################## RATES DUE TO ALL REACTIONS ############################


@njit
def change_solution_scalar(t, c_full, kact_sol, klig_sol, kloss_sol, khydro_sol, kcut_sol, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol):

    dc_dt = np.zeros(len(c_full))

    # changes due to reaction in solution
    dc_dt += change_activations_solution(c_full, kact_sol, acts_sol)
    dc_dt += change_ligations_solution(c_full, klig_sol, ligs_sol)
    dc_dt += change_losses_solution(c_full, kloss_sol, losses_sol)
    dc_dt += change_hydrolysis_solution(c_full, khydro_sol, hydros_sol)
    dc_dt += change_cleavages_solution(c_full, kcut_sol, cuts_sol)

    return dc_dt



@njit
def change_solution_kernel(t, c_full, Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol):

    dc_dt = np.zeros(len(c_full))

    # changes due to reactions in solution
    dc_dt += change_activations_solution_kernel(c_full, Kact_sol, acts_sol)
    dc_dt += change_ligations_solution_kernel(c_full, Klig_sol, ligs_sol)
    dc_dt += change_losses_solution_kernel(c_full, Kloss_sol, losses_sol)
    dc_dt += change_hydrolysis_solution_kernel(c_full, Khydro_sol, hydros_sol)
    dc_dt += change_cleavages_solution_kernel(c_full, Kcut_sol, cuts_sol)

    return dc_dt



@njit
def change_solution_kernel_and_template_scalar(t, c_full, \
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, \
    klig_temp, kloss_temp, khydro_temp, kcut_temp, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    ligs_temp, losses_temp, hydros_temp, cuts_temp2temp, cuts_temp2sol):

    dc_dt = np.zeros(len(c_full))

    # changes due to reactions in solution
    dc_dt += change_activations_solution_kernel(c_full, Kact_sol, acts_sol)
    dc_dt += change_ligations_solution_kernel(c_full, Klig_sol, ligs_sol)
    dc_dt += change_losses_solution_kernel(c_full, Kloss_sol, losses_sol)
    dc_dt += change_hydrolysis_solution_kernel(c_full, Khydro_sol, hydros_sol)
    dc_dt += change_cleavages_solution_kernel(c_full, Kcut_sol, cuts_sol)

    # changes due to reactions on template
    dc_dt += change_ligations_template_simplified(c_full, klig_temp, ligs_temp)
    dc_dt += change_losses_template_simplified(c_full, kloss_temp, losses_temp)
    dc_dt += change_hydrolysis_template_simplified(c_full, khydro_temp, hydros_temp)
    dc_dt += change_cleavage_temp2temp(c_full, kcut_temp, cuts_temp2temp)
    dc_dt += change_cleavage_temp2sol(c_full, kcut_temp, cuts_temp2sol)

    return dc_dt


######################## INTEGRATION OF TRAJECTORIES ###########################


def integrate_trajectory_solution_scalar(t_init, t_final, c_full_initial, \
    kact_sol, klig_sol, kloss_sol, khydro_sol, kcut_sol, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    table_sol, n_max):
    
    change_eff = partial(change_solution_scalar, \
        kact_sol=kact_sol, klig_sol=klig_sol, kloss_sol=kloss_sol, \
        khydro_sol=khydro_sol, kcut_sol=kcut_sol, \
        acts_sol=acts_sol, ligs_sol=ligs_sol, losses_sol=losses_sol, \
        hydros_sol=hydros_sol, cuts_sol=cuts_sol)
    
    # lists to store the integrated data
    ts = [t_init]
    cs = [c_full_initial]

    # integration
    RKint = integrate.RK45(change_eff, t_init, c_full_initial, t_final, rtol=1e-8, atol=1e-9)

    while(RKint.status == 'running'):
        RKint.step()
        ts.append(RKint.t)
        cs.append(RKint.y)
        #print("time: ", ts[-1])

    # computation of total oligomer concentration
    c_oligos, oligos_map = complexes.compute_oligomer_concentration_multitimes(cs, table_sol, n_max)

    # computation of total EDC concentration
    c_EDC = np.asarray(cs)[:,-1]

    return np.asarray(ts), c_oligos, oligos_map, c_EDC



def integrate_trajectory_solution_kernel(t_init, t_final, c_full_initial, \
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    table_sol, n_max):

    change_eff = partial(change_solution_kernel, \
        Kact_sol=Kact_sol, Klig_sol=Klig_sol, Kloss_sol=Kloss_sol, \
        Khydro_sol=Khydro_sol, Kcut_sol=Kcut_sol, \
        acts_sol=acts_sol, ligs_sol=ligs_sol, losses_sol=losses_sol, \
        hydros_sol=hydros_sol, cuts_sol=cuts_sol)

    # lists to store the integrated data
    ts = [t_init]
    cs = [c_full_initial]

    # integration
    RKint = integrate.RK45(change_eff, t_init, c_full_initial, t_final, rtol=1e-8, atol=1e-9)

    while(RKint.status == 'running'):
        RKint.step()
        ts.append(RKint.t)
        cs.append(RKint.y)
        #print("time: ", ts[-1])

    # computation of total oligomer concentration
    c_oligos, oligos_map = complexes.compute_oligomer_concentration_multitimes(cs, table_sol, n_max)

    # computation of total EDC concentration
    c_EDC = np.asarray(cs)[:,-1]

    return np.asarray(ts), c_oligos, oligos_map, c_EDC



def integrate_trajectory_solution_kernel_template_scalar(t_init, t_final, c_full_initial, \
    Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol, \
    klig_temp, kloss_temp, khydro_temp, kcut_temp, Kds_complex_full, Kds_complex_reduced, \
    acts_sol, ligs_sol, losses_sol, hydros_sol, cuts_sol, \
    ligs_temp, losses_temp, hydros_temp, cuts_temp2temp, cuts_temp2sol, \
    properties_complex_full, properties_complex_reduced, n_max):

    change_eff = partial(change_solution_kernel_and_template_scalar, \
        Kact_sol=Kact_sol, Klig_sol=Klig_sol, Kloss_sol=Kloss_sol, \
        Khydro_sol=Khydro_sol, Kcut_sol=Kcut_sol, \
        klig_temp=klig_temp, kloss_temp=kloss_temp, khydro_temp=khydro_temp, kcut_temp=kcut_temp, \
        acts_sol=acts_sol, ligs_sol=ligs_sol, losses_sol=losses_sol, \
        hydros_sol=hydros_sol, cuts_sol=cuts_sol, \
        ligs_temp=ligs_temp, losses_temp=losses_temp, hydros_temp=hydros_temp, \
        cuts_temp2temp=cuts_temp2temp, cuts_temp2sol=cuts_temp2sol)
    
    # equilibrate initial concentration
    c_full_initial_after_equ = compute_equilibrium_concentration(c_full_initial, \
        Kds_complex_full, Kds_complex_reduced, properties_complex_full, properties_complex_reduced)

    # lists to store the integrated data
    ts = [t_init]
    cs = [c_full_initial_after_equ]

    # integration
    while(ts[-1] <= t_final):
        RKint = integrate.RK45(change_eff, ts[-1], cs[-1], ts[-1]+0.5, rtol=1e-8, atol=1e-9)
        RKint.step()
        ts.append(RKint.t)
        c_before_equ = RKint.y
        c_after_equ = compute_equilibrium_concentration(c_before_equ, Kds_complex_full, \
            Kds_complex_reduced, properties_complex_full, properties_complex_reduced)
        #c_after_equ = c_before_equ
        cs.append(c_after_equ)
        print("time: ", ts[-1])


    # computation of total oligomer concentration
    c_oligos, oligos_map = complexes.compute_oligomer_concentration_multitimes_highperformance(\
        cs, properties_complex_full[2:], n_max)
    
    # computation of total EDC concentration
    c_EDC = np.asarray(cs)[:,-1]

    return np.asarray(ts), c_oligos, oligos_map, c_EDC, cs
