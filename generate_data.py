"""
optimization
"""

import math
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plot

import env
import RRT


class Optimization:
    def __init__(self, x_start, x_goal):
        self.env = env.Env()

        self.x_start = x_start
        self.x_goal = x_goal
        
        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
        self.N = 50
        self.M = 5
        
        self.initial_theta = 0
        self.terminal_theta = 0
        self.initial_phi = 0
        self.terminal_phi = 0
        self.initial_v = 0
        self.terminal_v = 0
        
        self.x_min = self.env.x_range[0]
        self.x_max = self.env.x_range[1]
        self.y_min = self.env.y_range[0]
        self.y_max = self.env.y_range[1]
        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.phi_min = -np.pi/4
        self.phi_max = np.pi/4
        self.v_min = 0
        self.v_max = 3
        
        self.robot_size = 0.5
        self.L = 1.5
        self.dt = 1
        
        self.set_cons = {'initial_x'     :True,                          #境界条件をセットするかどうか
                        'terminal_x'    :True, 
                        'initial_y'     :True, 
                        'terminal_y'    :True, 
                        'initial_theta' :True, 
                        'terminal_theta':True, 
                        'initial_phi'   :True, 
                        'terminal_phi'  :True,
                        'initial_v'     :True, 
                        'terminal_v'    :True}
        
        
        
    def generate_initial_path(self, rrt_path):
        N = self.N
        initial_theta = self.initial_theta 
        terminal_theta = self.terminal_theta 
        initial_phi = self.initial_phi 
        terminal_phi = self.terminal_phi 
        initial_v = self.initial_v 
        terminal_v = self.terminal_v
        dt = self.dt
        
        x, y = [], []
        for i in range(len(rrt_path)):
            x.append(rrt_path[i][0])
            y.append(rrt_path[i][1])
            
        tck, u = interpolate.splprep([x, y], k=3, s=0) 
        u = np.linspace(0, 1, num = N, endpoint = True)
        spline = interpolate.splev(u, tck)
        
        cubicX = spline[0]
        cubicY = spline[1]
        
        #nd.arrayに変換
        x = np.array(cubicX)
        y = np.array(cubicY)
        
        #x, yの差分を計算
        deltax = np.diff(x)
        deltay = np.diff(y)
        
        #x, y の差分からthetaを計算
        #theta[0]を初期値に置き換え、配列の最後に終端状態を追加
        theta = np.arctan(deltay / deltax)
        theta[0] = initial_theta
        theta = np.append(theta, terminal_theta)
        
        #thetaの差分からphiを計算
        #phi[0]を初期値に置き換え配列の最後に終端状態を追加
        deltatheta = np.diff(theta)
        phi = deltatheta / dt
        phi[0] = initial_phi
        phi = np.append(phi, terminal_phi)
        
        #x,yの差分からvを計算
        #phi[0]を初期値に置き換え配列の最後に終端状態を追加
        v = np.sqrt((deltax ** 2 + deltay ** 2) / dt)
        v[0] = initial_v
        v = np.append(v, terminal_v)
        
        return x, y, theta, phi, v
    
    def objective_function(self, x):
        N = self.N
        M = self.M
        
        phi_max = self.phi_max 
        v_max = self.v_max 
        
        #matrixに変換
        trajectory_matrix = x.reshape(M, N)
        
        #phiの二乗和を目的関数とする。
        sum = 0
        for i in range(N):
            sum += (trajectory_matrix[3, i] ** 2 / phi_max ** 2) + (trajectory_matrix[4, i] ** 2 / v_max ** 2) 
        
        return sum / N
    
    def generate_constraints(self):
        N = self.N
        obs_circle = self.obs_circle
        robot_size = self.robot_size
        L = self.L
        dt = self.dt
        set_cons = self.set_cons
        
        x_start = self.x_start
        x_goal = self.x_goal
        
        initial_theta = self.initial_theta 
        terminal_theta = self.terminal_theta 
        initial_phi = self.initial_phi 
        terminal_phi = self.terminal_phi 
        initial_v = self.initial_v 
        terminal_v = self.terminal_v
        
        #matrixに変換
        initial_x = x_start[0]
        initial_y = x_start[1]
        terminal_x = x_goal[0]
        terminal_y = x_goal[1]
        
        #最初に不等式制約(K×N個)
        cons = ()
        for k in range(len(obs_circle)):
            for i in range(N):
                cons = cons + ({'type':'ineq', 'fun':lambda x, i = i, k = k: ((x[i] - obs_circle[k][0]) ** 2 + (x[i + N] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + robot_size) ** 2},)

        #次にモデルの等式制約(3×(N-1)個)
        #x
        for i in range(N-1):
            cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1] - (x[i] + x[i + 4 * N] * np.cos(x[i + 2 * N]) * dt)},)
            
        #y
        for i in range(N-1):
            cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + N] - (x[i + N] + x[i + 4 * N] * np.sin(x[i + 2 * N]) * dt)},)
            
        #theta
        for i in range(N-1):
            cons = cons + ({'type':'eq', 'fun':lambda x, i = i: x[i+1 + 2 * N] - (x[i + 2 * N] + x[i + 4 * N] * np.tan(x[i+ 3 * N]) * dt / L)},)
            
            
        #境界条件(8個)
        #境界条件が設定されている場合は制約条件に加える。
        #x初期条件
        if set_cons['initial_x'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[0] - initial_x},)
            
        #x終端条件
        if set_cons['terminal_x'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[N - 1] - terminal_x},)

        #y初期条件
        if set_cons['initial_y'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[N] - initial_y},)
            
        #y終端条件
        if set_cons['terminal_y'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[2*N - 1] - terminal_y},)
            
        #theta初期条件
        if set_cons['initial_theta'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[2*N] - initial_theta},)
            
        #theta終端条件
        if set_cons['terminal_theta'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[3*N - 1] - terminal_theta},)
            
        #phi初期条件
        if set_cons['initial_phi'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[3*N] - initial_phi},)
            
        #phi終端条件
        if set_cons['terminal_phi'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[4*N - 1] - terminal_phi},)
            
        #v初期条件
        if set_cons['initial_v'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[4*N] - initial_v},)
            
        #v終端条件
        if set_cons['terminal_v'] == False:
            pass
        else:
            cons = cons + ({'type':'eq', 'fun':lambda x: x[5*N - 1] - terminal_v},)

        return cons
    
    
    def generate_bounds(self):
        N = self.N
        x_min = self.x_min 
        x_max = self.x_max 
        y_min = self.y_min 
        y_max = self.y_max 
        theta_min = self.theta_min 
        theta_max = self.theta_max 
        phi_min = self.phi_min 
        phi_max = self.phi_max 
        v_min = self.v_min 
        v_max = self.v_max
        
        #boundsのリストを生成
        bounds = []
        
        #xの範囲
        for i in range(N):
            bounds.append((x_min, x_max))
            
        #yの範囲
        for i in range(N):
            bounds.append((y_min, y_max))
            
        #thetaの範囲
        for i in range(N):
            bounds.append((theta_min, theta_max))
            
        #phiの範囲
        for i in range(N):
            bounds.append((phi_min, phi_max))
            
        #vの範囲
        for i in range(N):
            bounds.append((v_min, v_max))
            
        return bounds
    
    def generate_option(self):
        option = {'maxiter':1000}
        return option
    
    def optimize(self, initial_path):
        func = self.objective_function
        cons = self.generate_constraints
        bounds = self.generate_bounds
        options = self.generate_option
        
        return minimize(func, initial_path, method='SLSQP', constraints=cons, bounds=bounds)
    
def main():
    x_start = (2, 2)
    x_goal = (49, 24)
    
    #0.05の確率でゴールのノードをサンプリング
    rrt = RRT.Rrt(x_start, x_goal, 0.5, 0.05, 10000)
    path = rrt.planning()
    processed_path = rrt.utils.post_processing(path)
    if path:
        rrt.plotting.animation(rrt.vertex, path, "RRT", True)
        rrt.plotting.animation(rrt.vertex, processed_path, "RRT", False)
    else:
        print("No Path Found!")
    #ノードの順番を反転させる
    rrt_path = []
    for i in range(len(processed_path)):
        rrt_path.append(list(processed_path[-i-1]))
    
    optimization = Optimization(x_start, x_goal)
    initial_path = optimization.generate_initial_path(rrt_path)
    
    
    
    result = optimization.optimize(initial_path)
    
    print(result)
    plot.vis_env()
    plot.vis_path(initial_path)
    plot.compare_path(initial_path, result.x)
    plot.compare_history_theta(initial_path, result.x, range_flag = True)
    plot.compare_history_phi(initial_path, result.x, range_flag = True)
    plot.compare_history_v(initial_path, result.x, range_flag = True)
    plot.vis_history_theta(result.x, range_flag=True)
    plot.vis_history_phi(result.x, range_flag=True)
    plot.vis_history_v(result.x, range_flag = True)
    plot.compare_path_rec(initial_path, result.x)

if __name__ == '__main__':
    main()