import numpy as np
import gurobipy as gp

#CASO 1
'''
psoition of pickup and delivery points
The modification is needed since the original distance is represented by the time matrix
'''
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

loc = {} #Set of locations for each node first pickups then deliveries
q = {} #Set of demands, q in the model
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
q[0] = 0
q[1] = 1
q[2] = 1
q[3] = 1
q[4] = 1
q[5] = 1

q[6] = 1
q[7] = 1
q[8] = 1
q[9] = 1
q[10] = 1

q[11] = 1
q[12] = 1
q[13] = 1
q[14] = 1
q[15] = 1

q[16] = 1
q[17] = 1
q[18] = 1
q[19] = 1
q[20] = 1

q[21] = 1
q[22] = 1
q[23] = 1
q[24] = 1
q[25] = 1

q[26] = 1
q[27] = 1
q[28] = 1
q[29] = 1
q[30] = 1

q[31] = 1
q[32] = 1
q[33] = 1
q[34] = 1
q[35] = 1

q[36] = 1
q[37] = 1
q[38] = 1
q[39] = 1
q[40] = 1

q[41] = 1
q[42] = 1
q[43] = 1
q[44] = 1
q[45] = 1

q[46] = 1
q[47] = 1
q[48] = 1
q[49] = 1
q[50] = 1

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
q[51] = -1
q[52] = -1
q[53] = -1
q[54] = -1
q[55] = -1

q[56] = -1
q[57] = -1
q[58] = -1
q[59] = -1
q[60] = -1

q[61] = -1
q[62] = -1
q[63] = -1
q[64] = -1
q[65] = -1

q[66] = -1
q[67] = -1
q[68] = -1
q[69] = -1
q[70] = -1

q[71] = -1
q[72] = -1
q[73] = -1
q[74] = -1
q[75] = -1

q[76] = -1
q[77] = -1
q[78] = -1
q[79] = -1
q[80] = -1

q[81] = -1
q[82] = -1
q[83] = -1
q[84] = -1
q[85] = -1

q[86] = -1
q[87] = -1
q[88] = -1
q[89] = -1
q[90] = -1

q[91] = -1
q[92] = -1
q[93] = -1
q[94] = -1
q[95] = -1

q[96] = -1
q[97] = -1
q[98] = -1
q[99] = -1
q[100] = -1
q[101] = 0

n = 50 #Number of jobs
'''
descibe the sets of the pickup and delivery points
'''
P = [i for i in range(1, n + 1)] #Set of pickup nodes
D = [i for i in range(n + 1, 2*n + 1)] #Set opf delivery nodes
S = range(50) #Set of scenarios
N = [0] + P + D + [2*n + 1] #Set N from the paper

M = 1000 #Big M
ex = 0
'''
descibe the time of the pickup points at cycles 1 to 10 
'''
e = {}  # Pickup times, e in the model from paper
e[1] = 0
e[2] = 80
e[3] = 0
e[4] = 100
e[5] = 100

e[6] = 400
e[7] = 500
e[8] = 400
e[9] = 450
e[10] = 500

e[11] = 820
e[12] = 800
e[13] = 800
e[14] = 850
e[15] = 950

e[16] = 1200
e[17] = 1350
e[18] = 1200
e[19] = 1200
e[20] = 1300

e[21] = 1610
e[22] = 1700
e[23] = 1600
e[24] = 1680
e[25] = 1770

e[26] = 2010
e[27] = 2050
e[28] = 2000
e[29] = 2100
e[30] = 2100

e[31] = 2400
e[32] = 2430
e[33] = 2400
e[34] = 2470
e[35] = 2570

e[36] = 2800
e[37] = 2870
e[38] = 2800
e[39] = 2950
e[40] = 2900

e[41] = 3200
e[42] = 3210
e[43] = 3200
e[44] = 3370
e[45] = 3300

e[46] = 3630
e[47] = 3600
e[48] = 3600
e[49] = 3700
e[50] = 3750

for i in N:
    if i not in P:
        e[i] = 0

l = {}  # Delivery times, l in the model from paper
'''
descibe the time of the delivery points at cycles 1 to 10
'''
l[51] = 230+ex
l[52] = 310+ex
l[53] = 400+ex
l[54] = 330+ex
l[55] = 330+ex

l[56] = 630+ex
l[57] = 730+ex
l[58] = 800+ex
l[59] = 680+ex
l[60] = 730+ex

l[61] = 1050+ex
l[62] = 1030+ex
l[63] = 1200+ex
l[64] = 1080+ex
l[65] = 1180+ex

l[66] = 1430+ex
l[67] = 1580+ex
l[68] = 1600+ex
l[69] = 1430+ex
l[70] = 1530+ex

l[71] = 1840+ex
l[72] = 1930+ex
l[73] = 2000+ex
l[74] = 1910+ex
l[75] = 2000+ex

l[76] = 2240+ex
l[77] = 2280+ex
l[78] = 2400+ex
l[79] = 2330+ex
l[80] = 2330+ex

l[81] = 2630+ex
l[82] = 2660+ex
l[83] = 2800+ex
l[84] = 2700+ex
l[85] = 2800+ex

l[86] = 3030+ex
l[87] = 3100+ex
l[88] = 3200+ex
l[89] = 3180+ex
l[90] = 3130+ex

l[91] = 3430+ex
l[92] = 3440+ex
l[93] = 3600+ex
l[94] = 3600+ex
l[95] = 3530+ex

l[96] = 3860+ex
l[97] = 3830+ex
l[98] = 4000+ex
l[99] = 3930+ex
l[100] = 3980+ex


for i in N:
    if i not in D:
        l[i] = M

t = {} #traveling times
ts = {}
c = {} #distance "costs"
v = 1.5 #the speed of the AGV
mean = 1
var = 0.5
setuptime = 0
for i in N:
    for j in N:
        c[i,j] = round(np.linalg.norm(loc[i] - loc[j], 1), 2)
        for s in S:
            aux = round(max(0, np.random.normal(mean, var)) * c[i,j] / v, 2)
            #print(c[i,j]/v, aux)
            ts[i,j,s] = max(.1, aux)
        t[i,j] = max(0.1, round(c[i,j]/v,2) + setuptime)#np.mean([ts[i,j,s] for s in S])
        #print(t[i,j], np.mean([ts[i,j,s] for s in S]))

K = range(2) #Set of AGV available
Cap = {}
for k in K:
    Cap[k] = 1
'''
Using gurobi optimisation for the output the optimal results
'''

m = gp.Model() #Declaring model in Gurobi
m.Params.MIPFocus = 1 #Changing focus on the solver to integer
x = m.addVars(K, N, N, vtype = 'B', name = 'x') #x variables directs which AGV takes which job
B = m.addVars(K, N, vtype = 'C', name = 'w') #B variables are the time stamps
Q = m.addVars(K, N, vtype = 'C', name = 'q') #Q variables are the capacity conditions
ms = m.addVars(N, name = 'slack') #Slack variables for feasibility

m.addConstrs((gp.quicksum(x[k, i, j] for k in K for j in N) == 1
             for i in P), 'sources')
m.addConstrs((gp.quicksum(x[k, i, j] for j in N) - gp.quicksum(x[k, n+i, j] for j in N) == 0
             for i in P for k in K), 'nodes1')
m.addConstrs((gp.quicksum(x[k, j, i] for j in N) - gp.quicksum(x[k, i, j] for j in N) == 0
             for i in (P + D) for k in K), 'nodes2')
m.addConstrs((gp.quicksum(x[k, 0, j] for j in N) == 1
             for k in K), 'origin')
m.addConstrs((gp.quicksum(x[k, i, 2*n + 1] for i in N) == 1
             for k in K), 'terminal')
m.addConstrs((B[k, j] >= B[k, i] + t[i,j] - M * (1 - x[k, i, j])
             for i in N for j in N for k in K), 'job_times1')
m.addConstrs((Q[k, j] >= Q[k, i] + q[j] - M * (1 - x[k, i, j])
             for i in N for j in N for k in K), 'job_vols1')
m.addConstrs((B[k, i] + t[i, n+i] <= B[k, n+i]
             for i in P for k in K), 'job_prec')
m.addConstrs((B[k, i] >= e[i] - ms[i]
             for i in N for k in K), 'start_time')
m.addConstrs((B[k, i] <= l[i] + ms[i]
             for i in N for k in K), 'finish_time')
m.addConstrs((Q[k, i] >= max(0, q[j])
             for i in N for k in K), 'start_cap')
m.addConstrs((Q[k, i] <= min(Cap[k], Cap[k] + q[j])
             for i in N for k in K), 'finish_cap')
m.setObjective(gp.quicksum(c[i,j] * x[k,i,j] for k in K for i in N for j in N)
             + 1/M * (gp.quicksum(B) + gp.quicksum(Q)) + M * gp.quicksum(ms))

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
        for j in N:
            if x[k, i, j].x > 0:
                aux = j
        print(k, i, aux, round(B[k, i].X,2), round(B[k, aux].X,2))
        order[k].append(aux)
        if aux == i:
            aux = 2*n+1
        else:
            i = aux
    #order[k].append(0)