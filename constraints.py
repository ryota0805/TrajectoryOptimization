#不等式制約、等式制約を定義する
from param import Parameter as p
import util
import numpy as np

########
#制約条件を生成する関数
########
def generate_constraints():
    #最初に不等式制約
    cons = ()
    for i in range(p.N):
        cons = cons + ({'type':'ineq', 'fun':lambda x, i = i: ((x[i] - p.x_o) ** 2 + (x[i + p.N] - p.y_o) ** 2) - p.r_o ** 2},)

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

    return cons
########
#bounds(変数の範囲)を設定する関数
########

#変数の数だけタプルのリストとして返す関数
def generate_bounds():
    
    #boundsのリストを生成
    bounds = []
    
    #xの範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #yの範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #thetaの範囲
    for i in range(p.N):
        bounds.append((p.theta_min, p.theta_max))
        
    #phiの範囲
    for i in range(p.N):
        bounds.append((p.phi_min, p.phi_max))
        
    return bounds