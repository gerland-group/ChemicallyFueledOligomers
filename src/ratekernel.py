#!/bin/env python3

import numpy as np



def list_rate_constants():

    table_rate_constants = {}

    counter = 0

    # activation
    for n in range(1, 5):
        table_rate_constants["act_%s" %n] = counter
        counter += 1
    
    table_rate_constants["act_rest"] = counter
    counter += 1

    # ligation
    table_rate_constants["lig"] = counter
    counter += 1

    # loss
    table_rate_constants["loss"] = counter
    counter += 1

    # hydro
    table_rate_constants["hydro"] = counter
    counter += 1
    
    # cut
    for n in range(2, 5):
        table_rate_constants["cut_%s" %n] = counter
        counter += 1
    
    table_rate_constants["cut_rest"] = counter
    counter += 1
    
    return table_rate_constants



def choose_rate_constant_kernels(ks, table_sol, table_rate_constant, n_max):

    # activation rate constants
    Kact_sol = np.zeros(len(table_sol))

    for cmplx in table_sol.keys():
        if((cmplx != '') and (cmplx.split(',')[1] == '_') and (cmplx.split(',')[2] == '0')):
            # regular oligomers can get activated
            n = int(cmplx.split(',')[0])
            if(n<5):
                Kact_sol[table_sol[cmplx]] = ks[table_rate_constant['act_%s' %n]]
            else:
                Kact_sol[table_sol[cmplx]] = ks[table_rate_constant['act_rest']]
        else:
            continue

    # ligation rate constants
    Klig_sol = np.zeros((len(table_sol), len(table_sol)))

    complexes = list(table_sol.keys())

    for i in range(len(complexes)):
        for j in range(i+1, len(complexes)):
            cmplx1 = complexes[i]
            cmplx2 = complexes[j]
            if((cmplx1 != '') and (cmplx1.split(',')[1] == '_') \
                and (cmplx1.split(',')[2] == '0') \
                and (cmplx2 != '') and (cmplx2.split(',')[1] == '_') \
                and (cmplx2.split(',')[2] == '1')):

                n = int(cmplx1.split(',')[0])
                m = int(cmplx2.split(',')[0])
                if(n+m <= n_max):
                    Klig_sol[i,j] = ks[table_rate_constant['lig']]
                    Klig_sol[j,i] = ks[table_rate_constant['lig']]
            
            elif((cmplx1 != '') and (cmplx1.split(',')[1] == '_') \
                and (cmplx1.split(',')[2] == '1') \
                and (cmplx2 != '') and (cmplx2.split(',')[1] == '_') \
                and (cmplx2.split(',')[2] == '0')):

                n = int(cmplx1.split(',')[0])
                m = int(cmplx2.split(',')[0])
                if(n+m <= n_max):
                    Klig_sol[i,j] = ks[table_rate_constant['lig']]
                    Klig_sol[j,i] = ks[table_rate_constant['lig']]
    
    # loss rate constants
    Kloss_sol = np.zeros(len(table_sol))

    for cmplx in table_sol.keys():
        if((cmplx != '') and (cmplx.split(',')[1] == '_') and (cmplx.split(',')[2] == '1')):
            n = int(cmplx.split(',')[0])
            Kloss_sol[table_sol[cmplx]] = ks[table_rate_constant['loss']]
    
    # hydrolysis rate constants
    Khydro_sol = np.zeros(len(table_sol))

    for cmplx in table_sol.keys():
        if((cmplx != '') and (cmplx.split(',')[1] == '_') and (cmplx.split(',')[2] == '1')):
            n = int(cmplx.split(',')[0])
            Khydro_sol[table_sol[cmplx]] = ks[table_rate_constant['hydro']]
    
    # cleavage rate constants
    Kcut_sol = np.zeros(len(table_sol))

    for cmplx in table_sol.keys():
        if((cmplx != '') and (cmplx.split(',')[1] == '_')):
            n = int(cmplx.split(',')[0])
            if(n==1):
                continue
            elif(n<5):
                Kcut_sol[table_sol[cmplx]] = ks[table_rate_constant['cut_%s' %n]]
            else:
                Kcut_sol[table_sol[cmplx]] = ks[table_rate_constant['cut_rest']]

    return Kact_sol, Klig_sol, Kloss_sol, Khydro_sol, Kcut_sol



def compute_spanned_orders_of_magnitude(ks, n_max):

    # identify the rate constants
    ks_act = ks[0:5]
    ks_lig = ks[5] # scalar by definition
    ks_loss = ks[6] # scalar by definition
    ks_hydro = ks[7] # scalar by definition
    ks_cut = ks[8:]

    s_act = 1/(len(ks_act)-1) * np.sum((np.log10(ks_act[1:]/ks_act[:-1]))**2)
    s_cut = 1/(len(ks_cut)-1) * np.sum((np.log10(ks_cut[1:]/ks_cut[:-1]))**2)

    return s_act + s_cut
