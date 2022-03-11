import numpy as np
import gurobipy as gp

# CASO 1

NS = np.array([25, 10])
NA = np.array([0, 0])
NB = np.array([10, 0])
NC = np.array([0, 10])
ND = np.array([10, 15])
NE = np.array([25, -10])

loc = {}
d = {}  # Set of demands, q in the model
loc[0] = NS
loc[1] = NA
loc[2] = NC
loc[3] = NE
loc[4] = NB
loc[5] = NC
loc[6] = NE

d[0] = 0
d[1] = 1
d[2] = 1
d[3] = 1
d[4] = 1
d[5] = 1
d[6] = 1

loc[7] = NB
loc[8] = ND
loc[9] = NA
loc[10] = NA
loc[11] = ND
loc[12] = NC
loc[13] = NS

d[7] = -1
d[8] = -1
d[9] = -1
d[10] = -1
d[11] = -1
d[12] = -1
d[13] = 0

n = 6  # Number of jobs
P = [i for i in range(1, n + 1)]  # Set of pickup nodes
D = [i for i in range(n + 1, 2 * n + 1)]  # Set opf delivery nodes
S = range(50)  # Set of scenarios
V = [0] + P + D + [2 * n + 1]  # Set N from the paper

M = 1000  # Big M
ex = 0

a = {}  # Pickup times, e in the model from paper
a[1] = 10
a[2] = 20
a[3] = 10
a[4] = 20
a[5] = 40
a[6] = 20

for i in V:
    if i not in P:
        a[i] = 0

b = {}  # Delivery times, l in the model from paper
b[7] = 40 + ex
b[8] = 55 + ex
b[9] = 60 + ex
b[10] = 90 + ex
b[11] = 90 + ex
b[12] = 80 + ex

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