#!/bin/env python3

import numpy as np



def list_activations_solution(n_max, table_sol):
    
    # list activations in solution, i. e. conversion from regular oligomers 
    # to O-Acylisourea

    acts_sol_humanreadable = []

    for n in range(1, n_max+1):
        ins = ("%s,_,0" %n, "EDC")
        outs = ("%s,_,1" %n)
        acts_sol_humanreadable.append([ins, outs])

    ins = np.zeros((len(acts_sol_humanreadable), 2), dtype=int)
    outs = np.zeros((len(acts_sol_humanreadable), 1), dtype=int)

    for i, act_sol in enumerate(acts_sol_humanreadable):
        in1 = table_sol[act_sol[0][0]]
        ins[i] = np.array([in1, -1])

        out = table_sol[act_sol[1]]
        outs[i] = np.array([out])
    
    acts_sol = np.concatenate((ins, outs), axis=1)

    return acts_sol, acts_sol_humanreadable



def list_ligations_solution(n_max, table_sol):

    # list ligations in solution between regular oligomers and O-Acylisourea
    ligs_sol_humanreadable = []

    for n in range(1, n_max+1):
        for m in range(1, n_max+1):
            if(n+m <= n_max):
                ins = ("%s,_,0" %n, "%s,_,1" %m)
                outs =  ("%s,_,0" %(n+m))
                ligs_sol_humanreadable.append([ins, outs])

    ins = np.zeros((len(ligs_sol_humanreadable), 2), dtype=int)
    outs = np.zeros((len(ligs_sol_humanreadable), 1), dtype=int)

    for i, lig_sol in enumerate(ligs_sol_humanreadable):
        in1 = table_sol[lig_sol[0][0]]
        in2 = table_sol[lig_sol[0][1]]
        ins[i] = np.array([in1, in2])

        out = table_sol[lig_sol[1]]
        outs[i] = np.array([out])

    ligs_sol = np.concatenate((ins, outs), axis=1)

    return ligs_sol, ligs_sol_humanreadable



def list_losses_solution(n_max, table_sol):

    # list loss in solution
    losses_sol_humanreadable = []

    for n in range(1, n_max+1):
        ins = ("%s,_,1" %n)
        outs = ("%s,_,2" %n)
        losses_sol_humanreadable.append([ins, outs])

    ins = np.zeros((len(losses_sol_humanreadable), 1), dtype=int)
    outs = np.zeros((len(losses_sol_humanreadable), 1), dtype=int)

    for i, loss_sol in enumerate(losses_sol_humanreadable):
        in1 = table_sol[loss_sol[0]]
        ins[i] = np.array([in1])

        out = table_sol[loss_sol[1]]
        outs[i] = np.array([out])

    losses_sol = np.concatenate((ins, outs), axis=1)

    return losses_sol, losses_sol_humanreadable
 


def list_hydrolysis_solution(n_max, table_sol):

    # list hydrolysis in solution
    hydros_sol_humanreadable = []

    for n in range(1, n_max+1):
        ins = ("%s,_,1" %n)
        outs = ("%s,_,0" %n)
        hydros_sol_humanreadable.append([ins, outs])
    
    ins = np.zeros((len(hydros_sol_humanreadable), 1), dtype=int)
    outs = np.zeros((len(hydros_sol_humanreadable), 1), dtype=int)

    for i, hydro_sol in enumerate(hydros_sol_humanreadable):
        in1 = table_sol[hydro_sol[0]]
        ins[i] = np.array([in1])

        out = table_sol[hydro_sol[1]]
        outs[i] = np.array([out])

    hydros_sol = np.concatenate((ins, outs), axis=1)

    return hydros_sol, hydros_sol_humanreadable



def list_cleavages_solution(n_max, table_sol):

    # list cleavages in solution
    cuts_sol_humanreadable = []

    # cleavage of regular oligomers
    for n in range(2, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1

            ins = ("%s,_,0" %n)
            outs = ("%s,_,0" %m1, "%s,_,0" %m2)
            cuts_sol_humanreadable.append([ins, outs])

    # cleavage of activated oligomers (O-Acylisourea)
    for n in range(2, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1

            ins = ("%s,_,1" %n)
            outs = ("%s,_,0" %m1, "%s,_,1" %m2)
            cuts_sol_humanreadable.append([ins, outs])

    # cleavage of lost oligomers (N-Acylisourea)
    for n in range(2, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1

            ins = ("%s,_,2" %n)
            outs = ("%s,_,0" %m1, "%s,_,2" %m1)
            cuts_sol_humanreadable.append([ins, outs])

    ins = np.zeros((len(cuts_sol_humanreadable), 1), dtype=int)
    outs = np.zeros((len(cuts_sol_humanreadable), 2), dtype=int)

    for i, cut_sol in enumerate(cuts_sol_humanreadable):
        in1 = table_sol[cut_sol[0]]
        ins[i] = np.array([in1])

        out1 = table_sol[cut_sol[1][0]]
        out2 = table_sol[cut_sol[1][1]]
        outs[i] = np.array([out1,out2])

    cuts_sol = np.concatenate((ins, outs), axis=1)

    return cuts_sol, cuts_sol_humanreadable



def list_ligations_template_simplified(n_max, table_full):
    
    # list ligations between two regular oligomers under consumption of EDC on template
    # i. e. single-step reaction of activation and subsequent ligation
    # O-Acylisourea not modelled!

    ligs_temp_humanreadable = []

    complexes = list(table_full.keys())

    for cmplx in complexes:
        if("|" in cmplx):
            strands = cmplx.split('|')

            for i in range(len(strands)-1):
                strand1 = strands[i]
                n1 = int(strand1.split(',')[0])
                i1 = int(strand1.split(',')[1])
                c1 = int(strand1.split(',')[2])

                strand2 = strands[i+1]
                n2 = int(strand2.split(',')[0])
                i2 = int(strand2.split(',')[1])
                c2 = int(strand2.split(',')[2])

                if((i2 == i1+n1) and (n1+n2 <= n_max) and (c1 == 0) and (c2 == 0)):
                    cmplx_new = ""
                    for j in range(len(strands)):
                        if(j!=i and j!=i+1):
                            cmplx_new += (strands[j] + "|")
                        
                        elif(j==i):
                            cmplx_new += "%s,%s,%s|" %(n1+n2, i1, 0)
                    
                    cmplx_new = cmplx_new[0:-1]

                    ins = (cmplx, "EDC")
                    outs = (cmplx_new)

                    ligs_temp_humanreadable.append([ins, outs])

    ins = np.zeros((len(ligs_temp_humanreadable), 2), dtype=int)
    outs = np.zeros((len(ligs_temp_humanreadable), 1), dtype=int)

    for i, lig_temp in enumerate(ligs_temp_humanreadable):
        in1 = table_full[lig_temp[0][0]]
        ins[i] = np.array([in1, -1])

        out = table_full[lig_temp[1]]
        outs[i] = np.array([out])

    ligs_temp = np.concatenate((ins, outs), axis=1)

    return ligs_temp, ligs_temp_humanreadable



def list_losses_template_simplified(table_full):

    # conversion of regular oligomer into N-Acylisourea under consumption of EDC
    # instantaneous dehybridization of N-Acylisourea into solution

    losses_temp_humanreadable = []

    complexes = list(table_full.keys())

    for cmplx in complexes:
        if((cmplx != '') and (cmplx.split(',')[1] != '_')):
            # only considering proper complexes

            strands = cmplx.split('|')

            for j in range(len(strands)):
                strand = strands[j]
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c==0): # regular oligomer
                    ins = (cmplx, "EDC")

                    new_cmplx = ""
                    for k in range(len(strands)):
                        if(k != j):
                            new_cmplx += (strands[k] + "|")
                    new_cmplx = new_cmplx[0:-1]

                    new_strand = "%s,_,2" %n

                    outs = (new_cmplx, new_strand)

                    losses_temp_humanreadable.append([ins, outs])

    ins = np.zeros((len(losses_temp_humanreadable), 2), dtype=int)
    outs = np.zeros((len(losses_temp_humanreadable), 2), dtype=int)

    for i, loss_temp in enumerate(losses_temp_humanreadable):
        in1 = table_full[loss_temp[0][0]]
        ins[i] = np.array([in1, -1])

        out1 = table_full[loss_temp[1][0]]
        out2 = table_full[loss_temp[1][1]]
        outs[i] = np.array([out1, out2])

    losses_temp = np.concatenate((ins, outs), axis=1)

    return losses_temp, losses_temp_humanreadable



def list_hydrolysis_template_simplified(table_full):

    # list all hydrolysis reactions of oligomers on template
    # where a regular oligomer on the template gets activated (to O-Acylisourea)
    # and O-Acylisourea instantaneously hydrolyses, i. e. O-Acylisourea is 
    # not modelled explicitely

    hydros_temp_humanreadable = []

    complexes = list(table_full.keys())

    for cmplx in complexes:
        if((cmplx != '') and(cmplx.split(',')[1] != "_")):
            # only considering proper complexes

            strands = cmplx.split('|')

            for j in range(len(strands)):

                strand = strands[j]
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c==0):
                    ins = (cmplx, 'EDC')
                    outs = (cmplx)
                    hydros_temp_humanreadable.append([ins, outs])

    ins = np.zeros((len(hydros_temp_humanreadable), 2), dtype=int)
    outs = np.zeros((len(hydros_temp_humanreadable), 1), dtype=int)

    for i, hydro_temp in enumerate(hydros_temp_humanreadable):
        in1 = table_full[hydro_temp[0][0]]
        ins[i] = np.array([in1, -1])

        out1 = table_full[hydro_temp[1]]
        outs[i] = np.array([out1])

    hydros_temp = np.concatenate((ins, outs), axis=1)

    return hydros_temp, hydros_temp_humanreadable



def list_cleavages_temp2temp(Lt, degree_total, table_full):

    # list all cleavage reactions of oligomers on template
    # where the two products are located on the template after cleavage

    cuts_temp2temp_humanreadable = []

    complexes = list(table_full.keys())

    for cmplx in complexes:
        if((cmplx != '') and (cmplx.split(',')[1] != "_")):
            # only considering proper complexes

            strands = cmplx.split('|')

            # cleavage only possible if degree_complex <= degree_total-1

            if(len(strands) <= degree_total-1):

                for j in range(len(strands)):

                    strand = strands[j]
                    n = int(strand.split(',')[0])
                    i = int(strand.split(',')[1])
                    c = int(strand.split(',')[2])

                    if(c==0):
                        # regular oligomer gets cleaved
                        for m1 in range(1, n):
                            m2 = n-m1

                            # check if both child oligomers are located on the template
                            if((i+m1-1 >= 0) and (i+m1 < Lt)):
                                cmplx_new = ""
                                for k in range(len(strands)):
                                    if(k != j):
                                        cmplx_new += (strands[k] + "|")
                                    elif(k == j):
                                        cmplx_new += ("%s,%s,0|%s,%s,0|" %(m1,i,m2,i+m1))
                                cmplx_new = cmplx_new[0:-1]

                                # cuts_temp2temp_humanreadable.append([weight, ins, outs])
                                cuts_temp2temp_humanreadable.append([1, (cmplx), (cmplx_new)])
                
                    elif(c==1):
                        # O-Acylisourea gets cleaved
                        for m1 in range(1, n):
                            m2 = n-m1

                            # check if both child oligomers are located on the template
                            if((i+m1-1 >= 0) and (i+m1 < Lt)):
                                cmplx_new1 = ""
                                cmplx_new2 = ""
                                for k in range(len(strands)):
                                    if(k != j):
                                        cmplx_new1 += (strands[k] + "|")
                                        cmplx_new2 += (strands[k] + "|")
                                    elif(k == j):
                                        cmplx_new1 += ("%s,%s,0|%s,%s,1|" %(m1,i,m2,i+m1))
                                        cmplx_new2 += ("%s,%s,1|%s,%s,0|" %(m1,i,m2,i+m1))
                                cmplx_new1 = cmplx_new1[0:-1]
                                cmplx_new2 = cmplx_new2[0:-1]

                                cuts_temp2temp_humanreadable.append([0.5, (cmplx), (cmplx_new1)])
                                cuts_temp2temp_humanreadable.append([0.5, (cmplx), (cmplx_new2)])
                    
                    elif(c==2):
                        # N-Acylisourea gets cleaved
                        for m1 in range(1, n):
                            m2 = n-m1

                            # check if both child oligomers are located on the template
                            if((i+m1-1 >= 0) and (i+m1 < Lt)):
                                cmplx_new1 = ""
                                cmplx_new2 = ""
                                for k in range(len(strands)):
                                    if(k != j):
                                        cmplx_new1 += (strands[k] + "|")
                                        cmplx_new2 += (strands[k] + "|")
                                    elif(k == j):
                                        cmplx_new1 += ("%s,%s,0|%s,%s,2|" %(m1,i,m2,i+m1))
                                        cmplx_new2 += ("%s,%s,2|%s,%s,0|" %(m1,i,m2,i+m1))
                                cmplx_new1 = cmplx_new1[0:-1]
                                cmplx_new2 = cmplx_new2[0:-1]

                                cuts_temp2temp_humanreadable.append([0.5, (cmplx), (cmplx_new1)])
                                cuts_temp2temp_humanreadable.append([0.5, (cmplx), (cmplx_new2)])

    ws = np.zeros((len(cuts_temp2temp_humanreadable), 1))
    ins = np.zeros((len(cuts_temp2temp_humanreadable), 1), dtype=int)
    outs = np.zeros((len(cuts_temp2temp_humanreadable), 1), dtype=int)

    for i, cut_temp in enumerate(cuts_temp2temp_humanreadable):
        ws[i] = np.array([cut_temp[0]])
        
        in1 = table_full[cut_temp[1]]
        ins[i] = np.array([in1])

        out1 = table_full[cut_temp[2]]
        outs[i] = np.array([out1])

    cuts_temp2temp = np.concatenate((ws, ins, outs), axis=1)

    return cuts_temp2temp, cuts_temp2temp_humanreadable



def list_cleavages_temp2sol(Lt, table_full):

    # list all cleavage reactions of oligomers on a template
    # where the two products are located on the template after cleavage

    cuts_temp2sol_humanreadable = []

    complexes = list(table_full.keys())

    for cmplx in complexes:
        if((cmplx != '') and (cmplx.split(',')[1] != "_")):
            # only considering proper complexes

            strands = cmplx.split('|')

            for j in range(len(strands)):

                strand = strands[j]
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c == 0):
                    # regular oligomer
                    
                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new += ("%s,%s,0|" %(m2,i+m1))
                            cmplx_new = cmplx_new[0:-1]

                            ws = 1
                            ins = (cmplx)
                            outs = ("%s,_,0" %m1, cmplx_new)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs])

                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new += ("%s,%s,0|" %(m1,i))
                            cmplx_new = cmplx_new[0:-1]

                            ws = 1
                            ins = (cmplx)
                            outs = ("%s,_,0" %m2, cmplx_new)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs])

                elif(c==1):
                    # O-Acylisourea

                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,1|" %(m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            ws = 0.5
                            ins = (cmplx)
                            outs1 = ("%s,_,1" %m1, cmplx_new1)
                            outs2 = ("%s,_,0" %m1, cmplx_new2)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs1])
                            cuts_temp2sol_humanreadable.append([ws, ins, outs2])

                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m1,i))
                                    cmplx_new2 += ("%s,%s,1|" %(m1,i))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            ws = 0.5
                            ins = (cmplx)
                            outs1 = ("%s,_,1" %m2, cmplx_new1)
                            outs2 = ("%s,_,0" %m2, cmplx_new2)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs1])
                            cuts_temp2sol_humanreadable.append([ws, ins, outs2])

                elif(c==2):
                    # N-Acylisourea

                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,2|" %(m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            ws = 0.5
                            ins = (cmplx)
                            outs1 = ("%s,_,2" %m1, cmplx_new1)
                            outs2 = ("%s,_,0" %m1, cmplx_new2)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs1])
                            cuts_temp2sol_humanreadable.append([ws, ins, outs2])

                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m1,i))
                                    cmplx_new2 += ("%s,%s,2|" %(m1,i))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            ws = 0.5
                            ins = (cmplx)
                            outs1 = ("%s,_,2" %m2, cmplx_new1)
                            outs2 = ("%s,_,0" %m2, cmplx_new2)

                            cuts_temp2sol_humanreadable.append([ws, ins, outs1])
                            cuts_temp2sol_humanreadable.append([ws, ins, outs2])

    ws = np.zeros((len(cuts_temp2sol_humanreadable), 1))
    ins = np.zeros((len(cuts_temp2sol_humanreadable), 1), dtype=int)
    outs = np.zeros((len(cuts_temp2sol_humanreadable), 2), dtype=int)

    for i, cut_temp in enumerate(cuts_temp2sol_humanreadable):
        ws[i] = np.array([cut_temp[0]])
        
        in1 = table_full[cut_temp[1]]
        ins[i] = np.array([in1])

        out1 = table_full[cut_temp[2][0]]
        out2 = table_full[cut_temp[2][1]]
        outs[i] = np.array([out1, out2])
    
    cuts_temp2sol = np.concatenate((ws, ins, outs), axis=1)

    return cuts_temp2sol, cuts_temp2sol_humanreadable



def generate_table_complexes_solution_activation_productive_destructive(table_sol, n_max):
    
    table_complexes_solution_act_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_act_productive[table_sol[oligomer]] = []
    table_complexes_solution_act_productive[-1] = [] # for EDC

    table_complexes_solution_act_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_act_destructive[table_sol[oligomer]] = []
    table_complexes_solution_act_destructive[-1] = [] # for EDC

    for n in range(1, n_max+1):
        oligomer_in = "%s,_,%s" %(n,0)
        oligomer_out = "%s,_,%s" %(n,1)

        table_complexes_solution_act_productive[table_sol[oligomer_out]].append([table_sol[oligomer_in], -1])
        
        table_complexes_solution_act_destructive[table_sol[oligomer_in]].append([table_sol[oligomer_in], -1])
        table_complexes_solution_act_destructive[-1].append([table_sol[oligomer_in], -1])

    for oligomer in table_complexes_solution_act_productive.keys():

        table_complexes_solution_act_productive[oligomer] = \
            np.asarray(table_complexes_solution_act_productive[oligomer])
        table_complexes_solution_act_destructive[oligomer] = \
            np.asarray(table_complexes_solution_act_destructive[oligomer])

    return table_complexes_solution_act_productive, table_complexes_solution_act_destructive



def generate_table_complexes_solution_ligation_productive_destructive(table_sol, n_max):

    table_complexes_solution_ligation_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_ligation_productive[table_sol[oligomer]] = []
    table_complexes_solution_ligation_productive[-1] = [] # for EDC

    table_complexes_solution_ligation_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_ligation_destructive[table_sol[oligomer]] = []
    table_complexes_solution_ligation_destructive[-1] = [] # for EDC

    for n in range(1, n_max+1):
        for m in range(1, n_max+1):
            if(n+m<=n_max):
                oligomer_in1 = "%s,_,%s" %(n,0)
                oligomer_in2 = "%s,_,%s" %(m,1)
                oligomer_out = "%s,_,%s" %(n+m,0)

                table_complexes_solution_ligation_productive[table_sol[oligomer_out]].append(\
                    [table_sol[oligomer_in1], table_sol[oligomer_in2]])
                table_complexes_solution_ligation_destructive[table_sol[oligomer_in1]].append(\
                    [table_sol[oligomer_in1], table_sol[oligomer_in2]])
                table_complexes_solution_ligation_destructive[table_sol[oligomer_in2]].append(\
                    [table_sol[oligomer_in1], table_sol[oligomer_in2]])

    for oligomer in table_complexes_solution_ligation_productive.keys():

        table_complexes_solution_ligation_productive[oligomer] = \
            np.asarray(table_complexes_solution_ligation_productive[oligomer])
        table_complexes_solution_ligation_destructive[oligomer] = \
            np.asarray(table_complexes_solution_ligation_destructive[oligomer])

    return table_complexes_solution_ligation_productive, table_complexes_solution_ligation_destructive



def generate_table_complexes_solution_loss_productive_destructive(table_sol, n_max):

    table_complexes_solution_loss_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_loss_productive[table_sol[oligomer]] = []
    table_complexes_solution_loss_productive[-1] = []

    table_complexes_solution_loss_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_loss_destructive[table_sol[oligomer]] = []
    table_complexes_solution_loss_destructive[-1] = []

    for n in range(1, n_max+1):
        oligomer_in = "%s,_,%s" %(n,1)
        oligomer_out = "%s,_,%s" %(n,2)

        table_complexes_solution_loss_productive[table_sol[oligomer_out]].append(table_sol[oligomer_in])
        table_complexes_solution_loss_destructive[table_sol[oligomer_in]].append(table_sol[oligomer_in])
    
    for oligomer in table_complexes_solution_loss_productive.keys():

        table_complexes_solution_loss_productive[oligomer] = \
            np.asarray(table_complexes_solution_loss_productive[oligomer])
        table_complexes_solution_loss_destructive[oligomer] = \
            np.asarray(table_complexes_solution_loss_destructive[oligomer])

    return table_complexes_solution_loss_productive, table_complexes_solution_loss_destructive



def generate_table_complexes_solution_hydro_productive_destructive(table_sol, n_max):

    table_complexes_solution_hydro_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_hydro_productive[table_sol[oligomer]] = []
    table_complexes_solution_hydro_productive[-1] = []

    table_complexes_solution_hydro_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_hydro_destructive[table_sol[oligomer]] = []
    table_complexes_solution_hydro_destructive[-1] = []

    for n in range(1, n_max+1):
        oligomer_in = "%s,_,%s" %(n,1)
        oligomer_out = "%s,_,%s" %(n,0)

        table_complexes_solution_hydro_productive[table_sol[oligomer_out]].append(table_sol[oligomer_in])
        table_complexes_solution_hydro_destructive[table_sol[oligomer_in]].append(table_sol[oligomer_in])

    for oligomer in table_complexes_solution_hydro_productive.keys():

        table_complexes_solution_hydro_productive[oligomer] = \
            np.asarray(table_complexes_solution_hydro_productive[oligomer])
        table_complexes_solution_hydro_destructive[oligomer] = \
            np.asarray(table_complexes_solution_hydro_destructive[oligomer])

    return table_complexes_solution_hydro_productive, table_complexes_solution_hydro_destructive



def generate_table_complexes_solution_cleavage_productive_destructive(table_sol, n_max):

    table_complexes_solution_cleavage_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_cleavage_productive[table_sol[oligomer]] = []
    table_complexes_solution_cleavage_productive[-1] = [] # for EDC

    table_complexes_solution_cleavage_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_solution_cleavage_destructive[table_sol[oligomer]] = []
    table_complexes_solution_cleavage_destructive[-1] = [] # for EDC

    # for regular oligomers    
    for n in range(1, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1
            oligomer_in = "%s,_,%s" %(n,0)
            oligomer_out1 = "%s,_,%s" %(m1,0)
            oligomer_out2 = "%s,_,%s" %(m2,0)

            table_complexes_solution_cleavage_productive[table_sol[oligomer_out1]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_productive[table_sol[oligomer_out2]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_destructive[table_sol[oligomer_in]].append(table_sol[oligomer_in])

    # for activated oligomers
    for n in range(1, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1
            oligomer_in = "%s,_,%s" %(n,1)
            oligomer_out1 = "%s,_,%s" %(m1,0)
            oligomer_out2 = "%s,_,%s" %(m2,1)

            table_complexes_solution_cleavage_productive[table_sol[oligomer_out1]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_productive[table_sol[oligomer_out2]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_destructive[table_sol[oligomer_in]].append(table_sol[oligomer_in])
    
    # for lost oligomers
    for n in range(1, n_max+1):
        for m1 in range(1, n):
            m2 = n-m1
            oligomer_in = "%s,_,%s" %(n,2)
            oligomer_out1 = "%s,_,%s" %(m1,0)
            oligomer_out2 = "%s,_,%s" %(m2,2)

            table_complexes_solution_cleavage_productive[table_sol[oligomer_out1]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_productive[table_sol[oligomer_out2]].append(table_sol[oligomer_in])
            table_complexes_solution_cleavage_destructive[table_sol[oligomer_in]].append(table_sol[oligomer_in])
    
    for oligomer in table_complexes_solution_cleavage_productive.keys():

        table_complexes_solution_cleavage_productive[oligomer] = \
            np.asarray(table_complexes_solution_cleavage_productive[oligomer])
        table_complexes_solution_cleavage_destructive[oligomer] = \
            np.asarray(table_complexes_solution_cleavage_destructive[oligomer])

    return table_complexes_solution_cleavage_productive, table_complexes_solution_cleavage_destructive



def generate_table_complexes_templated_ligation_productive_destructive_restricted(\
    table_sol, table_full):

    table_complexes_templated_ligation_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_ligation_productive[table_sol[oligomer]] = []
    table_complexes_templated_ligation_productive[-1] = []

    table_complexes_templated_ligation_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_ligation_destructive[table_sol[oligomer]] = []
    table_complexes_templated_ligation_destructive[-1] = []

    for cmplx in table_full.keys():
        if((cmplx != '') and (len(cmplx.split('|'))>1)):
            # complexes that contain at least two oligomers
            strands = cmplx.split('|')

            for i in range(len(strands)-1):

                # left strand
                strand1 = strands[i]
                n1 = int(strand1.split(',')[0])
                i1 = int(strand1.split(',')[1])
                c1 = int(strand1.split(',')[2])

                # right strand
                strand2 = strands[i+1]
                n2 = int(strand2.split(',')[0])
                i2 = int(strand2.split(',')[1])
                c2 = int(strand2.split(',')[2])

                if((i2 == i1+n1) and (c1==0) and (c2==0)):
                    # construct the complex that is produced in the ligation
                    # to check if it is a valid complex
                    
                    cmplx_new = ""

                    for j in range(len(strands)):
                        if(j!=i and j!=i+1):
                            cmplx_new += (strands[j] + "|")
                        
                        elif(j==i):
                            cmplx_new += "%s,%s,%s|" %(n1+n2, i1, 0)

                    cmplx_new = cmplx_new[0:-1]

                    if(cmplx_new in table_full):
                        # produced complex is considered to be valid

                        # identify the oligomers involved in reaction
                        oligomer_in1 = "%s,_,%s" %(n1, c1)
                        oligomer_in2 = "%s,_,%s" %(n2, c2)
                        oligomer_out = "%s,_,%s" %(n1+n2,0)

                        # add oligomers to table of reactive oligomers
                        table_complexes_templated_ligation_destructive[\
                            table_sol[oligomer_in1]].append([table_full[cmplx], -1])
                        table_complexes_templated_ligation_destructive[\
                            table_sol[oligomer_in2]].append([table_full[cmplx], -1])
                        table_complexes_templated_ligation_destructive[-1].append([table_full[cmplx], -1])

                        table_complexes_templated_ligation_productive[\
                            table_sol[oligomer_out]].append([table_full[cmplx], -1])

    for oligomer in table_complexes_templated_ligation_productive.keys():

        table_complexes_templated_ligation_productive[oligomer] = \
            np.asarray(table_complexes_templated_ligation_productive[oligomer])
        table_complexes_templated_ligation_destructive[oligomer] = \
            np.asarray(table_complexes_templated_ligation_destructive[oligomer])
    
    return table_complexes_templated_ligation_productive, table_complexes_templated_ligation_destructive



def generate_table_complexes_templated_loss_productive_destructive(table_sol, \
    table_full):

    table_complexes_templated_loss_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_loss_productive[table_sol[oligomer]] = []
    table_complexes_templated_loss_productive[-1] = []

    table_complexes_templated_loss_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_loss_destructive[table_sol[oligomer]] = []
    table_complexes_templated_loss_destructive[-1] = []

    for cmplx in table_full.keys():
        if((cmplx != '') and (cmplx.split(',')[1] != '_')):
            # only strands that are part of a complex allowed; no strands in solution
            strands = cmplx.split('|')

            for j, strand in enumerate(strands):
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c==0):
                    # create new cmplx to check if it is contained in the full table
                    # only then the reaction is performed
                    new_cmplx = ""
                    for k in range(len(strands)):
                        if(k != j):
                            new_cmplx += (strands[k] + "|")
                    new_cmplx = new_cmplx[0:-1]

                    if(new_cmplx in table_full):
                        # reaction is allowed
                        oligomer_before = "%s,_,%s" %(n,c)
                        oligomer_after = "%s,_,%s" %(n,2)
                        
                        table_complexes_templated_loss_destructive[table_sol[oligomer_before]].append([table_full[cmplx], -1])
                        table_complexes_templated_loss_destructive[-1].append([table_full[cmplx], -1])

                        table_complexes_templated_loss_productive[table_sol[oligomer_after]].append([table_full[cmplx], -1])

    for oligomer in table_complexes_templated_loss_productive.keys():

        table_complexes_templated_loss_productive[oligomer] = \
            np.asarray(table_complexes_templated_loss_productive[oligomer])
        table_complexes_templated_loss_destructive[oligomer] = \
            np.asarray(table_complexes_templated_loss_destructive[oligomer])
    
    return table_complexes_templated_loss_productive, table_complexes_templated_loss_destructive



def generate_table_complexes_templated_cleavage_temp2temp_productive_destructive_restricted(\
    table_sol, table_full, Lt):

    table_complexes_templated_cleavage_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_cleavage_productive[table_sol[oligomer]] = []
    table_complexes_templated_cleavage_productive[-1] = []

    table_complexes_templated_cleavage_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_cleavage_destructive[table_sol[oligomer]] = []
    table_complexes_templated_cleavage_destructive[-1] = []    

    for cmplx in table_full.keys():
        if((cmplx != '') and (cmplx.split(',')[1] != '_')):
            # only considering propert complexes

            strands = cmplx.split('|')

            for j in range(len(strands)):

                strand = strands[j]
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c==0):
                    # regular oligomer gets cleaved
                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if both child oligomers are located on the template
                        if((i+m1-1 >= 0) and (i+m1 < Lt)):
                            cmplx_new = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new += ("%s,%s,0|%s,%s,0|" %(m1,i,m2,i+m1))
                            cmplx_new = cmplx_new[0:-1]

                            # check if the generated complex is part of the table
                            if(cmplx_new in table_full):
                                
                                oligomer_in = "%s,_,%s" %(n,0)
                                oligomer_out1 = "%s,_,%s" %(m1,0)
                                oligomer_out2 = "%s,_,%s" %(m2,0)
                                
                                # first position in each entry corresponds to a weight
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out1]].append([1, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out2]].append([1, table_full[cmplx]])

                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])
                        
                elif(c==1):
                    # O-Acylisouread gets cleaved
                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if both child oligomers are located on the template
                        if((i+m1-1 >= 0) and (i+m1 < Lt)):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|%s,%s,1|" %(m1,i,m2,i+m1))
                                    cmplx_new2 += ("%s,%s,1|%s,%s,0|" %(m1,i,m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            # check if the generated complexes are part of the table
                            if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):
                                
                                oligomer_in = "%s,_,%s" %(n,1)
                                oligomer_out_11 = "%s,_,%s" %(m1,0)
                                oligomer_out_12 = "%s,_,%s" %(m2,1)
                                oligomer_out_21 = "%s,_,%s" %(m1,1)
                                oligomer_out_22 = "%s,_,%s" %(m2,0)
                                
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_11]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_12]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_21]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_22]].append([0.5, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])
                
                elif(c==2):
                    # N-Acylisouread gets cleaved
                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if both child oligomers are locaed on the template
                        if((i+m1-1 >= 0) and (i+m1 < Lt)):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|%s,%s,2|" %(m1,i,m2,i+m1))
                                    cmplx_new2 += ("%s,%s,2|%s,%s,0|" %(m1,i,m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):

                                oligomer_in = "%s,_,%s" %(n,2)
                                oligomer_out_11 = "%s,_,%s" %(m1,0)
                                oligomer_out_12 = "%s,_,%s" %(m2,2)
                                oligomer_out_21 = "%s,_,%s" %(m1,2)
                                oligomer_out_22 = "%s,_,%s" %(m2,0)

                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_11]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_12]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_21]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out_22]].append([0.5, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])

    for oligomer in table_complexes_templated_cleavage_productive.keys():

        table_complexes_templated_cleavage_productive[oligomer] = \
            np.asarray(table_complexes_templated_cleavage_productive[oligomer])
        table_complexes_templated_cleavage_destructive[oligomer] = \
            np.asarray(table_complexes_templated_cleavage_destructive[oligomer])

    return table_complexes_templated_cleavage_productive, table_complexes_templated_cleavage_destructive
        


def generate_table_complexes_templated_cleavage_temp2sol_productive_destructive_restricted(\
    table_sol, table_full, Lt):

    table_complexes_templated_cleavage_productive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_cleavage_productive[table_sol[oligomer]] = []
    table_complexes_templated_cleavage_productive[-1] = []

    table_complexes_templated_cleavage_destructive = {}
    for oligomer in table_sol.keys():
        table_complexes_templated_cleavage_destructive[table_sol[oligomer]] = []
    table_complexes_templated_cleavage_destructive[-1] = []    

    for cmplx in table_full.keys():
        if((cmplx != '') and (cmplx.split(',')[1] != '_')):
            # only considering propert complexes

            strands = cmplx.split('|')

            for j in range(len(strands)):

                strand = strands[j]
                n = int(strand.split(',')[0])
                i = int(strand.split(',')[1])
                c = int(strand.split(',')[2])

                if(c==0):
                    # regular oligomer

                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new += (strands[k] + "|")
                                elif(k ==j):
                                    cmplx_new += ("%s,%s,0|" %(m2,i+m1))
                            cmplx_new = cmplx_new[0:-1]
                        
                            if(cmplx_new in table_full):
                                
                                oligomer_in = "%s,_,%s" %(n,0)
                                oligomer_out1 = "%s,_,%s" %(m1,0)
                                oligomer_out2 = "%s,_,%s" %(m2,0)

                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out1]].append([1, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out2]].append([1, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])

                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new += ("%s,%s,0|" %(m1,i))
                            cmplx_new = cmplx_new[0:-1]

                            if(cmplx_new in table_full):

                                oligomer_in = "%s,_,%s" %(n,0)
                                oligomer_out1 = "%s,_,%s" %(m1,0)
                                oligomer_out2 = "%s,_,%s" %(m2,0)

                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out1]].append([1, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out2]].append([1, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])
                            
                elif(c==1):
                    # O-Acylisourea

                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,1|" %(m2,i+m1))
                                cmplx_new1 = cmplx_new1[0:-1]
                                cmplx_new2 = cmplx_new2[0:-1]

                                if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):
                                    # both complexes exist

                                    oligomer_in = "%s,_,%s" %(n,1)
                                    oligomer_out11 = "%s,_,%s" %(m1,1)
                                    oligomer_out12 = "%s,_,%s" %(m2,0)
                                    oligomer_out21 = "%s,_,%s" %(m1,0)
                                    oligomer_out22 = "%s,_,%s" %(m2,1)

                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out11]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out12]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out21]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out22]].append([0.5, table_full[cmplx]])
                                    
                                    table_complexes_templated_cleavage_destructive[\
                                        table_sol[oligomer_in]].append([1, table_full[cmplx]])
                                    
                        
                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,1|" %(m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):
                                # both complexes exist

                                oligomer_in = "%s,_,%s" %(n,1)
                                oligomer_out11 = "%s,_,%s" %(m1,0)
                                oligomer_out12 = "%s,_,%s" %(m2,1)
                                oligomer_out21 = "%s,_,%s" %(m1,1)
                                oligomer_out22 = "%s,_,%s" %(m2,0)

                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out11]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out12]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out21]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out22]].append([0.5, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])
                
                elif(c==2):
                    # N-Acylisourea

                    for m1 in range(1, n):
                        m2 = n-m1

                        # check if left child oligomer is in solution
                        if(i+m1-1 < 0):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,2|" %(m2,i+m1))
                                cmplx_new1 = cmplx_new1[0:-1]
                                cmplx_new2 = cmplx_new2[0:-1]

                                if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):
                                    # both complexes exist

                                    oligomer_in = "%s,_,%s" %(n,2)
                                    oligomer_out11 = "%s,_,%s" %(m1,2)
                                    oligomer_out12 = "%s,_,%s" %(m2,0)
                                    oligomer_out21 = "%s,_,%s" %(m1,0)
                                    oligomer_out22 = "%s,_,%s" %(m2,2)

                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out11]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out12]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out21]].append([0.5, table_full[cmplx]])
                                    table_complexes_templated_cleavage_productive[\
                                        table_sol[oligomer_out22]].append([0.5, table_full[cmplx]])
                                    
                                    table_complexes_templated_cleavage_destructive[\
                                        table_sol[oligomer_in]].append([1, table_full[cmplx]])
                                    
                        
                        # check if right child oligomer is in solution
                        if(i+m1 >= Lt):
                            cmplx_new1 = ""
                            cmplx_new2 = ""
                            for k in range(len(strands)):
                                if(k != j):
                                    cmplx_new1 += (strands[k] + "|")
                                    cmplx_new2 += (strands[k] + "|")
                                elif(k == j):
                                    cmplx_new1 += ("%s,%s,0|" %(m2,i+m1))
                                    cmplx_new2 += ("%s,%s,2|" %(m2,i+m1))
                            cmplx_new1 = cmplx_new1[0:-1]
                            cmplx_new2 = cmplx_new2[0:-1]

                            if((cmplx_new1 in table_full) and (cmplx_new2 in table_full)):
                                # both complexes exist

                                oligomer_in = "%s,_,%s" %(n,2)
                                oligomer_out11 = "%s,_,%s" %(m1,0)
                                oligomer_out12 = "%s,_,%s" %(m2,2)
                                oligomer_out21 = "%s,_,%s" %(m1,2)
                                oligomer_out22 = "%s,_,%s" %(m2,0)

                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out11]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out12]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out21]].append([0.5, table_full[cmplx]])
                                table_complexes_templated_cleavage_productive[\
                                    table_sol[oligomer_out22]].append([0.5, table_full[cmplx]])
                                
                                table_complexes_templated_cleavage_destructive[\
                                    table_sol[oligomer_in]].append([1, table_full[cmplx]])

    for oligomer in table_complexes_templated_cleavage_productive.keys():

        table_complexes_templated_cleavage_productive[oligomer] = \
            np.asarray(table_complexes_templated_cleavage_productive[oligomer])
        table_complexes_templated_cleavage_destructive[oligomer] = \
            np.asarray(table_complexes_templated_cleavage_destructive[oligomer])

    return table_complexes_templated_cleavage_productive, table_complexes_templated_cleavage_destructive
