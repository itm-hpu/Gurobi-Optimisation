import numpy as np
import gurobipy as gp

#CASO 1
'''
psoition of pickup and delivery points
The modification is needed since the original distance is represented by the time matrix
'''
# NS = np.array([25, 10])
# NA = np.array([0, 0])
# NB = np.array([10, 0])
# NC = np.array([0, 10])
# ND = np.array([10, 15])
# NE = np.array([25, -10])
NA=np.array([7.00,10.00])
NB=np.array([7.20,6.00])
NC=np.array([9.20,6.00])
ND=np.array([7.20,7.50])
NE=np.array([9.20,7.50])
NF=np.array([7.20,9.00])
NG=np.array([9.20,9.00])
NH=np.array([16.00,8.00])
NI=np.array([15.00,6.00])
NJ=np.array([13.00,6.60])
NK=np.array([13.60,9.30])
NL=np.array([10.60,9.30])

'''
descibe the task using the mapping of two loc[] and 'd' is se
The modification is needed since the original position points are different.
'''

loc = {}
d = {} #Set of demands, q in the model

'''
the definition of From Location
'''
loc[0]=NA
loc[1]=NA
loc[2]=NH
loc[3]=NK
loc[4]=NH
loc[5]=NH
loc[6]=NC
loc[7]=NA
loc[8]=NH
loc[9]=NH
loc[10]=NE

loc[11]=NA
loc[12]=NA
loc[13]=NA
loc[14]=NH
loc[15]=NH

loc[16]=NA
loc[17]=NC
loc[18]=NL
loc[19]=NE
loc[20]=NG

loc[21]=NH
loc[22]=NH
loc[23]=NK
loc[24]=NH
loc[25]=NA

loc[26]=NC
loc[27]=NE
loc[28]=NI
loc[29]=NH
loc[30]=NG

loc[31]=NA
loc[32]=NA
loc[33]=NJ
loc[34]=NH
loc[35]=NH

loc[36]=NC
loc[37]=NE
loc[38]=NL
loc[39]=NC
loc[40]=NG

loc[41]=NH
loc[42]=NH
loc[43]=NJ
loc[44]=NH
loc[45]=NA

loc[46]=NH
loc[47]=NG
loc[48]=NI
loc[49]=NC
loc[50]=NE
'''
define the number of the each task, two nodes connected are defined as a task.
'''
d[0] = 0
d[1] = 0
d[2] = 1
d[3] = 1
d[4] = 1
d[5] = 1

d[6] = 1
d[7] = 1
d[8] = 1
d[9] = 1
d[10] = 1

d[11] = 1
d[12] = 1
d[13] = 1
d[14] = 1
d[15] = 1

d[16] = 1
d[17] = 1
d[18] = 1
d[19] = 1
d[20] = 1

d[21] = 1
d[22] = 1
d[23] = 1
d[24] = 1
d[25] = 1

d[26] = 1
d[27] = 1
d[28] = 1
d[29] = 1
d[30] = 1

d[31] = 1
d[32] = 1
d[33] = 1
d[34] = 1
d[35] = 1

d[36] = 1
d[37] = 1
d[38] = 1
d[39] = 1
d[40] = 1

d[41] = 1
d[42] = 1
d[43] = 1
d[44] = 1
d[45] = 1

d[46] = 1
d[47] = 1
d[48] = 1
d[49] = 1
d[50] = 1

'''
the definition of To Location
'''

loc[51]=NH
loc[52]=NB
loc[53]=NJ
loc[54]=ND
loc[55]=NF

loc[56]=NA
loc[57]=NH
loc[58]=NI
loc[59]=NF
loc[60]=NA

loc[61]=NB
loc[62]=NH
loc[63]=NI
loc[64]=NB
loc[65]=ND

loc[66]=NH
loc[67]=NA
loc[68]=NI
loc[69]=NA
loc[70]=NA

loc[71]=NB
loc[72]=ND
loc[73]=NJ
loc[74]=NF
loc[75]=NH

loc[76]=NA
loc[77]=NA
loc[78]=NK
loc[79]=NB
loc[80]=NA

loc[81]=NH
loc[82]=NH
loc[83]=NA
loc[84]=NB
loc[85]=ND

loc[86]=NA
loc[87]=NA
loc[88]=NK
loc[89]=NJ
loc[90]=NA

loc[91]=NB
loc[92]=NF
loc[93]=NL
loc[94]=NF
loc[95]=NH

loc[96]=ND
loc[97]=NA
loc[98]=NL
loc[99]=NA
loc[100]=NA
loc[101] = NA
'''
define the number of the each task
'''
d[51] = -1
d[52] = -1
d[53] = -1
d[54] = -1
d[55] = -1

d[56] = -1
d[57] = -1
d[58] = -1
d[59] = -1
d[60] = -1

d[61] = -1
d[62] = -1
d[63] = -1
d[64] = -1
d[65] = -1

d[66] = -1
d[67] = -1
d[68] = -1
d[69] = -1
d[70] = -1

d[71] = -1
d[72] = -1
d[73] = -1
d[74] = -1
d[75] = -1

d[76] = -1
d[77] = -1
d[78] = -1
d[79] = -1
d[80] = -1

d[81] = -1
d[82] = -1
d[83] = -1
d[84] = -1
d[85] = -1

d[86] = -1
d[87] = -1
d[88] = -1
d[89] = -1
d[90] = -1

d[91] = -1
d[92] = -1
d[93] = -1
d[94] = -1
d[95] = -1

d[96] = -1
d[97] = -1
d[98] = -1
d[99] = -1
d[100] = -1
d[101] = 0

n = 50 #Number of jobs
'''
descibe the sets of the pickup and delivery points
'''
P = [i for i in range(1, n + 1)]  # Set of pickup nodes
D = [i for i in range(n + 1, 2 * n + 1)]  # Set opf delivery nodes
S = range(50)  # Set of scenarios
V = [0] + P + D + [2 * n + 1]  # Set N from the paper

M = 1000  # Big M
ex = 0
'''
descibe the time of the pickup points at cycles 1 to 10 
'''
a = {}  # Pickup times, e in the model from paper
a[1] = 0
a[2] = 80
a[3] = 0
a[4] = 100
a[5] = 100

a[6] = 400
a[7] = 500
a[8] = 400
a[9] = 450
a[10] = 500

a[11] = 820
a[12] = 800
a[13] = 800
a[14] = 850
a[15] = 950

a[16] = 1200
a[17] = 1350
a[18] = 1200
a[19] = 1200
a[20] = 1300

a[21] = 1610
a[22] = 1700
a[23] = 1600
a[24] = 1680
a[25] = 1770

a[26] = 2010
a[27] = 2050
a[28] = 2000
a[29] = 2100
a[30] = 2100

a[31] = 2400
a[32] = 2430
a[33] = 2400
a[34] = 2470
a[35] = 2570

a[36] = 2800
a[37] = 2870
a[38] = 2800
a[39] = 2950
a[40] = 2900

a[41] = 3200
a[42] = 3210
a[43] = 3200
a[44] = 3370
a[45] = 3300

a[46] = 3630
a[47] = 3600
a[48] = 3600
a[49] = 3700
a[50] = 3750

for i in V:
    if i not in P:
        a[i] = 0

b = {}  # Delivery times, l in the model from paper
'''
descibe the time of the delivery points at cycles 1 to 10
'''
b[51] = 230+ex
b[52] = 310+ex
b[53] = 400+ex
b[54] = 330+ex
b[55] = 330+ex

b[56] = 630+ex
b[57] = 730+ex
b[58] = 800+ex
b[59] = 680+ex
b[60] = 730+ex

b[61] = 1050+ex
b[62] = 1030+ex
b[63] = 1200+ex
b[64] = 1080+ex
b[65] = 1180+ex

b[66] = 1430+ex
b[67] = 1580+ex
b[68] = 1600+ex
b[69] = 1430+ex
b[70] = 1530+ex

b[71] = 1840+ex
b[72] = 1930+ex
b[73] = 2000+ex
b[74] = 1910+ex
b[75] = 2000+ex

b[76] = 2240+ex
b[77] = 2280+ex
b[78] = 2400+ex
b[79] = 2330+ex
b[80] = 2330+ex

b[81] = 2630+ex
b[82] = 2660+ex
b[83] = 2800+ex
b[84] = 2700+ex
b[85] = 2800+ex

b[86] = 3030+ex
b[87] = 3100+ex
b[88] = 3200+ex
b[89] = 3180+ex
b[90] = 3130+ex

b[91] = 3430+ex
b[92] = 3440+ex
b[93] = 3600+ex
b[94] = 3600+ex
b[95] = 3530+ex

b[96] = 3860+ex
b[97] = 3830+ex
b[98] = 4000+ex
b[99] = 3930+ex
b[100] = 3980+ex

for i in V:
    if i not in D:
        b[i] = M

t = {}  # traveling times
ts = {}
c = {}  # distance "costs"
v = 1.5  # the speed of the AGV
mean = 1
var = 0.5
setuptime = 0
for i in V:
    for j in V:
        c[i, j] = round(np.linalg.norm(loc[i] - loc[j], 1), 2)
        for s in S:
            aux = round(max(0, np.random.normal(mean, var)) * c[i, j] / v, 2)
            # print(c[i,j]/v, aux)
            ts[i, j, s] = max(.1, aux)
        t[i, j] = max(0.1, round(c[i, j] / v, 2) + setuptime)  # np.mean([ts[i,j,s] for s in S])
        # print(t[i,j], np.mean([ts[i,j,s] for s in S]))

K = range(2)  # Set of AGV available
Cap = {}
for k in K:
    Cap[k] = 1
'''
Using gurobi optimisation for the output the optimal results
'''

m = gp.Model()
m.Params.MIPFocus = 1
x = m.addVars(K, V, V, vtype = 'B', name = 'x')
w = m.addVars(K, V, vtype = 'C', name = 'w')
q = m.addVars(K, V, vtype = 'C', name = 'q')
ms = m.addVars(V, name = 'slack')

m.addConstrs((gp.quicksum(x[k, i, j] for k in K for j in V) == 1
             for i in P), 'sources')
m.addConstrs((gp.quicksum(x[k, i, j] for j in V) - gp.quicksum(x[k, n+i, j] for j in V) == 0
             for i in P for k in K), 'nodes1')
m.addConstrs((gp.quicksum(x[k, j, i] for j in V) - gp.quicksum(x[k, i, j] for j in V) == 0
             for i in (P + D) for k in K), 'nodes2')
m.addConstrs((gp.quicksum(x[k, 0, j] for j in V) == 1
             for k in K), 'origin')
m.addConstrs((gp.quicksum(x[k, i, 2*n + 1] for i in V) == 1
             for k in K), 'terminal')
m.addConstrs((w[k, j] >= w[k, i] + t[i,j] - M * (1 - x[k, i, j])
             for i in V for j in V for k in K), 'job_times1')
m.addConstrs((q[k, j] >= q[k, i] + d[j] - M * (1 - x[k, i, j])
             for i in V for j in V for k in K), 'job_vols1')
m.addConstrs((w[k, i] + t[i, n+i] <= w[k, n+i]
             for i in P for k in K), 'job_prec')
m.addConstrs((w[k, i] >= a[i] - ms[i]
             for i in V for k in K), 'start_time')
m.addConstrs((w[k, i] <= b[i] + ms[i]
             for i in V for k in K), 'finish_time')
m.addConstrs((q[k, i] >= max(0, d[j])
             for i in V for k in K), 'start_cap')
m.addConstrs((q[k, i] <= min(Cap[k], Cap[k] + d[j])
             for i in V for k in K), 'finish_cap')
m.setObjective(gp.quicksum(c[i,j] * x[k,i,j] for k in K for i in V for j in V)
             + 1/M * (gp.quicksum(w) + gp.quicksum(q)) + M * gp.quicksum(ms))

m.write('model-pdptw-det.lp')

m.optimize()
m.write('bots-det_%s_%s.sol' % (n, len(K)))
for i in gp.tuplelist(ms):
    if round(ms[i].X, 0) > 0:
        print(ms[i])
order = {}
for k in K:
    order[k] = [0]
    i = 0
    aux = 0
    while aux < (2*n + 1):
        for j in V:
            if x[k, i, j].x > 0:
                aux = j
        print(k, i, aux, round(w[k, i].X,2), round(w[k, aux].X,2))
        order[k].append(aux)
        if aux == i:
            aux = 2*n+1
        else:
            i = aux
    #order[k].append(0)
fails = {}
import copy
orderd = copy.deepcopy(order)
for k in K:
    fails[k] = []
    if len(order[k]) > 2:
        for s in range(1000):
            true_w = 0
            for i,r in enumerate(order[k]):
                if r in P:
                    true_w = max(true_w, a[r])
                    #print(k, r, a[r], true_w)
                    if a[r] > true_w:
                        fails[k].append(1)
                    else:
                        fails[k].append(0)
                if r in D:
                    aux = round(max(0, np.random.normal(mean, var)) * c[i,j] / v, 2)
                    true_w += round(aux , 2)
                    #print(k, r, b[r], true_w, aux, order[k][i-1], c[order[k][i-1], r])
                    if b[r] < true_w:
                        fails[k].append(1)
                    else:
                        fails[k].append(0)
for k in K:
    if len(fails[k]) > 0:
        print(k, np.mean(fails[k]))
    else:
        print(k, 0)
xdet = {}
for i in gp.tuplelist(x):
    xdet[i] = int(x[i].X)

alpha = 0
m = gp.Model()
x = m.addVars(K, V, V, vtype='B', name='x')
w = m.addVars(K, V, S, vtype='C', name='w')
q = m.addVars(K, V, vtype='C', name='q')
z = m.addVars(S, vtype='B', name='z')
ms = m.addVar(name='makespan')

m.addConstrs((gp.quicksum(x[k, i, j] for k in K for j in V) == 1
              for i in P), 'sources')
m.addConstrs((gp.quicksum(x[k, i, j] for j in V) - gp.quicksum(x[k, n + i, j] for j in V) == 0
              for i in P for k in K), 'nodes1')
m.addConstrs((gp.quicksum(x[k, j, i] for j in V) - gp.quicksum(x[k, i, j] for j in V) == 0
              for i in (P + D) for k in K), 'nodes2')
m.addConstrs((gp.quicksum(x[k, 0, j] for j in V) == 1
              for k in K), 'origin')
m.addConstrs((gp.quicksum(x[k, i, 2 * n + 1] for i in V) == 1
              for k in K), 'terminal')
m.addConstrs((w[k, j, s] >= w[k, i, s] + ts[i, j, s] - M * (1 - x[k, i, j]) - M * z[s]
              for i in V for j in V for k in K for s in S), 'job_times1')
m.addConstrs((q[k, j] >= q[k, i] + d[j] - M * (1 - x[k, i, j])
              for i in V for j in V for k in K), 'job_vols1')
m.addConstrs((w[k, i, s] + ts[i, n + i, s] <= w[k, n + i, s] + M * z[s]
              for i in P for k in K for s in S), 'job_prec')
m.addConstrs((w[k, i, s] >= a[i] - a[i] * z[s] - ms
              for i in P for k in K for s in S), 'start_time')
m.addConstrs((w[k, i, s] <= b[i] + M * z[s] + ms
              for i in D for k in K for s in S), 'finish_time')
m.addConstrs((q[k, i] >= max(0, d[j])
              for i in P for k in K), 'start_cap')
m.addConstrs((q[k, i] <= min(Cap[k], Cap[k] + d[j])
              for i in P for k in K), 'finish_cap')
# m.addConstrs((w[k, i, s] <= ms[s] + M * z[s]
#             for k in K for i in V for s in S), 'max_makespan')
m.addConstr(gp.quicksum(z) <= alpha * len(S))
m.setObjective(gp.quicksum(c[i, j] * x[k, i, j] for k in K for i in V for j in V) + 1 / M * (
            gp.quicksum(w) + gp.quicksum(q)) + M * ms)
# m.addConstrs((gp.quicksum(x[k, i, j] for i in P for j in V) <= 1.2 * np.ceil(n/len(K))
#             for k in K), 'work_load')

m.write('model-pdptw-cc.lp')
for i in gp.tuplelist(x):
    x[i].start = xdet[i]
m.optimize()
m.write('bots-cc_%s_%s.sol' % (n, len(K)))
#print(1/len(S) * sum(ms[i].X for i in gp.tuplelist(ms)))
order = {}
for k in K:
    order[k] = [0]
    i = 0
    aux = 0
    while aux < (2*n + 1):
        for j in V:
            if x[k, i, j].x > 0:
                aux = j
        print(k, i, aux)
        order[k].append(aux)
        if aux == i:
            aux = 2*n+1
        else:
            i = aux
    #order[k].append(0)
fails = {}
for k in K:
    fails[k] = []
    if len(order[k]) > 2:
        for s in range(1000):
            true_w = 0
            for i,r in enumerate(order[k]):
                if r in P:
                    true_w = max(true_w, a[r])
                    #print(k, r, a[r], true_w)
                    if a[r] > true_w:
                        fails[k].append(1)
                    else:
                        fails[k].append(0)
                if r in D:
                    aux = round(max(0, np.random.normal(mean, var)) * c[i,j] / v, 2)
                    true_w += round(aux , 2)
                    #print(k, r, b[r], true_w)
                    if b[r] < true_w:
                        fails[k].append(1)
                    else:
                        fails[k].append(0)
for k in K:
    if len(fails[k]) > 0:
        print(k, np.mean(fails[k]))
    else:
        print(k, 0)
for k in K:
    print(orderd)
print(gp.quicksum(c[i,j] * xdet[k,i,j] for k in K for i in V for j in V))
print("----------------------------")
for k in K:
    print(order)
print(gp.quicksum(c[i,j] * x[k,i,j].X for k in K for i in V for j in V))

