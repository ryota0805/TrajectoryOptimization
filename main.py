#mainファイル

########
#import
########
from param import Parameter as p
import GenerateInitialPath
import util
import constraints
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import objective_function 
import plot


########
#WayPointから設計変数の初期値を計算する
########
cubicX, cubicY = GenerateInitialPath.cubic_spline()
x, y, theta, phi = GenerateInitialPath.gengerate_initialpath(cubicX, cubicY)

trajectory_matrix = np.array([x, y, theta, phi])

trajectory_vector = util.matrix_to_vector(trajectory_matrix)
'''
function = util.evaluate_function(trajectory_vector)
const = util.generate_constraints()
boundary = util.generate_bounds()
'''
def f(x):
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += trajectory_matrix[3, i] ** 2
    
    return sum / p.N



#制約条件を作成
#最初に不等式制約
cons = ()
for k in range(len(p.obstacle_list)):
    for i in range(p.N):
        cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: ((x[i] - p.obstacle_list[k][0]) ** 2 + (x[i + p.N] - p.obstacle_list[k][1]) ** 2) - p.obstacle_list[k][2] ** 2},)

#次にモデルの等式制約
#x
for i in range(p.N-1):
    cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1] - (x[i] + p.v * np.cos(x[i + 2 * p.N]) * p.dt)},)
    
#y
for i in range(p.N-1):
    cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + p.N] - (x[i + p.N] + p.v * np.sin(x[i + 2 * p.N]) * p.dt)},)
    
#theta
for i in range(p.N-1):
    cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + 2 * p.N] - (x[i + 2 * p.N] + p.v * np.tan(x[i+ 3 * p.N]) * p.dt / p.L)},)

#境界条件
cons = cons + ({'type':'eq', 'fun':lambda x: x[0] - p.initial_x},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N - 1] - p.terminal_x},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[p.N] - p.initial_y},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N - 1] - p.terminal_y},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[2*p.N] - p.initial_theta},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N - 1] - p.terminal_theta},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[3*p.N] - p.initial_phi},)
cons = cons + ({'type':'eq', 'fun':lambda x: x[4*p.N - 1] - p.terminal_phi},)


print(len(cons))

#consts = constraints.generate_constraints()
bounds = constraints.generate_bounds()
options = {'maxiter':1000}


func = objective_function.objective_function
result = optimize.minimize(func, trajectory_vector, method='SLSQP', constraints = cons, bounds = bounds, options = options)
print(result)

plot.compare_path(trajectory_vector, result.x)
plot.vis_history_theta(result.x, range_flag=True)
plot.vis_history_phi(result.x, range_flag=True)






