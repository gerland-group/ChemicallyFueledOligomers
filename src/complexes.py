#!/bin/env python3

import numpy as np
from numba import njit

# import own modules
import integrator



@njit
def classifier(cs):
    degree_O = len(np.where(cs==1)[0])
    degree_N = len(np.where(cs==2)[0])
    return np.array([degree_O, degree_N])



def generate_table_full_complexity(n_max, Lt, degree_total, degree_O, degree_N):

    # bounds on maximum number of O-Acylisourea and N-Acylisourea compounds in complex
    degree_bounds = np.array([degree_O, degree_N])

    table_sol = {}
    table_temp = {}

    counter=0

    # template only
    table_sol[''] = counter
    counter += 1

    # loop over simplices 
    # regular strands in solution (c=0)
    # activated strands in solution (= O-Acylisourea, c=1)
    # lost strands in solution (= N-Acylisourea, c=2)
    for n1 in range(1, n_max+1):
        for c1 in range(0, 3, 1):
            table_sol["%s,_,%s" %(n1,c1)] = counter
            counter += 1

    if(degree_total == 0):
        return table_sol

    counter = 0

    # loop over all duplexes containing only regular strands (c=0)
    for n1 in range(1, n_max+1):
        for i1 in range(-(n1-1),Lt):
            for c1 in range(0, 3, 1):
                if(np.all(classifier(np.array([c1])) <= degree_bounds)):
                    table_temp["%s,%s,%s" %(n1,i1,c1)] = counter
                    counter+=1

    table_temp_list = [table_temp]

    for d in range(1, degree_total):
        
        counter = 0
        table_temp_new = {}

        for cmplx_old in table_temp_list[d-1].keys():
            
            strands = cmplx_old.split('|')
            cs = np.asarray([int(strand.split(',')[2]) for strand in strands])
            
            strand_last = strands[-1]
            n_last = int(strand_last.split(',')[0])
            i_last = int(strand_last.split(',')[1])
            c_last = int(strand_last.split(',')[2])

            for n_new in range(1, n_max+1):
                for i_new in range(i_last+n_last, Lt):
                    for c_new in range(0, 3, 1):
                        if(np.all(classifier(np.concatenate((cs, np.array([c_new])))) \
                            <= degree_bounds)):
                            cmplx_new = cmplx_old + "|%s,%s,%s" %(n_new,i_new,c_new)
                            table_temp_new[cmplx_new] = counter
                            counter += 1
        
        table_temp_list.append(table_temp_new)

    # combine the tables
    table = {}

    counter = 0
    for cmplx in table_sol.keys():
        table[cmplx] = counter
        counter += 1

    for table_temp in table_temp_list:
        for cmplx in table_temp.keys():
            table[cmplx] = counter
            counter += 1

    return table



def compute_oligomer_concentration_singletime(c_full, table_full, n_max):

    # array to store the oligomer concentration
    c_oligos = np.zeros(3*n_max)

    # compute oligomer concentration
    for cmplx in table_full:
        if(cmplx != ''):
            if(not "|" in cmplx):
                # single strand in solution or only one strand hybridized to template
                n = int(cmplx.split(',')[0])
                c = int(cmplx.split(',')[2])
                c_oligos[3*(n-1)+c] += c_full[table_full[cmplx]]

            else:
                strands = cmplx.split('|')
                for strand in strands:
                    n = int(strand.split(',')[0])
                    c = int(strand.split(',')[2])
                    c_oligos[3*(n-1)+c] += c_full[table_full[cmplx]]
        
        else:
            # bare template
            continue

    return c_oligos




def compute_oligomer_concentration_multitimes(cs, table_full, n_max):

    # array to store the oligomer concentrations
    cs_oligos = np.zeros((3*n_max, len(cs)))

    for it in range(len(cs)):
        cs_oligos[:,it] = compute_oligomer_concentration_singletime(cs[it], table_full, n_max)

    # dictionary of indices
    oligos_map = {}

    for n in range(1, n_max+1):
        oligos_map["%s,_,0" %n] = 3*(n-1)
        oligos_map["%s,_,1" %n] = 3*(n-1)+1
        oligos_map["%s,_,2" %n] = 3*(n-1)+2

    return cs_oligos, oligos_map



def compute_oligomer_concentration_multitimes_highperformance(cs, ps, n_max):

    # array to store the oligomer concentrations
    cs_oligos = np.zeros((3*n_max, len(cs)))

    for it in range(len(cs)):
        cs_oligos[:,it] = integrator.compute_total_oligomer_concentration(cs[it], ps)[1:] 
        # dropped the first entry as it is the template concentration

    # dictionary of indices
    oligos_map = {}

    for n in range(1, n_max+1):
        oligos_map["%s,_,0" %n] = 3*(n-1)
        oligos_map["%s,_,1" %n] = 3*(n-1)+1
        oligos_map["%s,_,2" %n] = 3*(n-1)+2
    
    return cs_oligos, oligos_map



@njit
def compute_length_hybridization_site(n, i, Lt):
    if(i>=Lt):
        raise ValueError("invalid length of hybridization site!")
    
    else:
        hyb_site = min(i+n, Lt-i, n, Lt)
        return hyb_site



def generate_table_reduced_complexity(table_full, Lt):

    table_reduced = {}
    
    counter = -1
    
    for key_full in table_full.keys():

        if(key_full == ''):
            counter += 1
            table_reduced[key_full] = counter
            

        elif((key_full != '') and (key_full.split(',')[1] == '_')):
            n = int(key_full.split(',')[0])
            c = int(key_full.split(',')[2])
            if('%s,0,%s' %(n,c) not in table_reduced):    
                counter += 1
                table_reduced['%s,0,%s' %(n,c)] = counter

        else:
            key_new_table = {}

            strands = key_full.split('|')
            for strand in strands:
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])
                l = compute_length_hybridization_site(n, i, Lt)
                try:
                    key_new_table[(n,l,c)] += 1
                except:
                    key_new_table[(n,l,c)] = 1

            keys_sorted = sorted(key_new_table)
            
            key_new = ""
            for key in keys_sorted:
                for n in range(key_new_table[key]):
                    key_new += "%s,%s,%s|" %(key[0],key[1],key[2])
            
            key_new = key_new[0:-1]

            if(key_new not in table_reduced):
                counter += 1
                table_reduced[key_new] = counter

    return table_reduced



def generate_map_reduced_full(table_full, table_reduced, Lt):

    map_reduced_full = {}

    for key_full in table_full.keys():

        if(key_full == ''):
            key_reduced = key_full
            try:
                map_reduced_full[table_reduced[key_reduced]].append([table_full[key_full]])
            except:
                map_reduced_full[table_reduced[key_reduced]] = [table_full[key_full]]

        elif((key_full != '') and (key_full.split(',')[1] == '_')):
            n = int(key_full.split(',')[0])
            c = int(key_full.split(',')[2])
            
            key_reduced = '%s,0,%s' %(n,c)
            try:
                map_reduced_full[table_reduced[key_reduced]].append(table_full[key_full])
            except:
                map_reduced_full[table_reduced[key_reduced]] = [table_full[key_full]]

        else:
            key_reduced_table = {}

            strands = key_full.split('|')
            for strand in strands:
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])
                l = compute_length_hybridization_site(n, i, Lt)
                try:
                    key_reduced_table[(n,l,c)] += 1
                except:
                    key_reduced_table[(n,l,c)] = 1

            entries_sorted = sorted(key_reduced_table)
            
            key_reduced = ""
            for entry in entries_sorted:
                for n in range(key_reduced_table[entry]):
                    key_reduced += "%s,%s,%s|" %(entry[0],entry[1],entry[2])
            
            key_reduced = key_reduced[0:-1]

            try:
                map_reduced_full[table_reduced[key_reduced]].append(table_full[key_full])
            except:
                map_reduced_full[table_reduced[key_reduced]] = [table_full[key_full]]
    
    return map_reduced_full



def list_affiliations_oligomers_reduced_complexity(table_sol_reduced, table_reduced, \
    map_reduced_full):

    affiliation_oligomer_complex = {}
    
    for key in table_sol_reduced.keys():
        affiliation_oligomer_complex[key] = {}
    
    for index, key_reduced in enumerate(table_reduced):

        if(key_reduced == ''):
            key_sol_reduced = key_reduced
            affiliation_oligomer_complex[key_sol_reduced][key_reduced] = 1

        elif((key_reduced != '') and (key_reduced.split(',')[1] == '0')):
            n = int(key_reduced.split(',')[0])
            c = int(key_reduced.split(',')[2])
            key_sol_reduced = '%s,0,%s' %(n,c)
            affiliation_oligomer_complex[key_sol_reduced][key_reduced] = 1
            
        else:
            w1 = len(map_reduced_full[table_reduced[key_reduced]])

            affiliation_oligomer_complex[''][key_reduced] = w1

            strands = key_reduced.split('|')
            for strand in strands:
                n = int(strand.split(',')[0])
                c = int(strand.split(',')[2])
                
                key_sol_reduced = "%s,0,%s" %(n,c)

                try:
                    affiliation_oligomer_complex[key_sol_reduced][key_reduced] += w1
                except:
                    affiliation_oligomer_complex[key_sol_reduced][key_reduced] = w1
                
    return affiliation_oligomer_complex



def construct_complex_properties_reduced_complexity(table_sol_reduced, \
    affiliation_oligomer_complex):

    properties_list = []
    
    for oligomer in table_sol_reduced:
        
        properties1 = np.zeros((len(affiliation_oligomer_complex[oligomer]), 3))
        properties2 = np.zeros((len(affiliation_oligomer_complex[oligomer]), len(table_sol_reduced)))

        for j, cmplx in enumerate(affiliation_oligomer_complex[oligomer]):

            if(cmplx in table_sol_reduced):
                properties2[j, table_sol_reduced[cmplx]] = 1
                properties1[j,0] = affiliation_oligomer_complex[oligomer][cmplx] # weight
                properties1[j,1] = 0 # order of complex
                properties1[j,2] = 0 # length of hybridization site
        
            else:
                properties1[j,0] = affiliation_oligomer_complex[oligomer][cmplx] # weight

                properties2[j, table_sol_reduced['']] = 1 # template contained in the complex

                strands = cmplx.split('|')

                for strand in strands:

                    n = int(strand.split(',')[0])
                    l = int(strand.split(',')[1])
                    c = int(strand.split(',')[2])

                    properties2[j, table_sol_reduced['%s,0,%s' %(n,c)]] += 1
                    properties1[j,1] += 1 # order of complex
                    properties1[j,2] += l # length of hybridization site
            
        properties = np.concatenate((properties1, properties2), axis=1)

        properties_list.append(properties)
    
    return tuple(properties_list)



def construct_complex_properties_full_complexity(table_sol, table_full, Lt):

    cmplxs = list(table_full.keys())

    ps = np.zeros((len(table_sol), len(table_full)))
    ls = np.zeros((len(table_full)))
    ds = np.zeros((len(table_full)))

    for j, cmplx in enumerate(cmplxs):
        if(cmplx in table_sol):
            ps[table_sol[cmplx],j] = 1
            ls[j] = 0 
            ds[j] = 0

        else:
            ps[0,j] = 1 # template contained in the complex
            
            strands = cmplx.split('|')
            
            for strand in strands:
                
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])
                l = compute_length_hybridization_site(n, i, Lt)

                ps[table_sol["%s,_,%s" %(n,c)],j] += 1
                ls[j] += l
                ds[j] += 1

    properties = np.concatenate((np.array([ds]), np.array([ls]), ps), axis=0)

    return properties



def construct_complex_Kds_reduced_complexity(Kds, table_sol_reduced, \
    affiliation_oligomer_complex):

    Kdprods_list = []

    for oligomer in table_sol_reduced:

        Kdprods = np.ones(len(affiliation_oligomer_complex[oligomer]))

        for j, cmplx in enumerate(affiliation_oligomer_complex[oligomer]):

            if(cmplx in table_sol_reduced):
                Kdprods[j] = 1
            
            else:
                strands = cmplx.split('|')

                for strand in strands:

                    n = int(strand.split(',')[0])
            
                    Kdprods[j] *= Kds[n-1]
            
        Kdprods_list.append(Kdprods)

    return tuple(Kdprods_list)



def construct_complex_Kds_full_complexity(Kds, table_sol, table_full, Lt):

    cmplxs = list(table_full.keys())

    Kdprods = np.ones(len(table_full))

    for j, cmplx in enumerate(cmplxs):
        if(cmplx in table_sol):
            continue
    
        else:
            strands = cmplx.split('|')

            for strand in strands:

                n = int(strand.split(',')[0])
                Kdprods[j] *= Kds[n-1]

    return Kdprods
