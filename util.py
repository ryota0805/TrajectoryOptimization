from param import Parameter as p
import numpy as np

########
#設計変数の行列(M×N)をベクトル(1×MN)に変換する関数
########
def matrix_to_vector(trajectory_matrix):
    
    trajectory_vector = trajectory_matrix.flatten()
    
    return trajectory_vector

########
#設計変数のベクトル(1×MN)を行列(M×N)をに変換する関数
########
def vector_to_matrix(trajectory_vector):
    
    trajectory_matrix = trajectory_vector.reshape(p.M, p.N)
    
    return trajectory_matrix

########
#目的関数f(x)
########
def evaluate_function(trajectory_vector):
    #matrixに変換
    trajectory_matrix = trajectory_vector.reshape(p.M, p.N)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += trajectory_matrix[3, i] ** 2
    
    return sum / p.N
    
    
########
#不等式制約h(x)
########

#障害物の数KとWayPointの数Nの積の数KNだけ不等式制約を生成。辞書に追加
def generate_ineq_constraints():
    
    #ineq_constraintsを保存する辞書の生成
    ineq_constraints_dict = {}
    
    #各障害物に対して、各WayPointの制約関数を生成し、辞書に追加
    for k in range(p.K):
        x_k, y_k, r_k = p.obstacle_list[k][0], p.obstacle_list[k][1], p.obstacle_list[k][2]
        
        for i in range(p.N):
            
            #不等式制約の定義
            def ineq_constraint(trajectory_vector):
                trajectory_matrix = vector_to_matrix(trajectory_vector)
                
                return ((trajectory_matrix[0, i] - x_k) ** 2 + (trajectory_matrix[1, i] - y_k) ** 2) - r_k ** 2
            
            #ineq_constraints_dictに定義した関数を追加
            ineq_constraints_dict['ineq_constraints'+'_'+str(k)+'_'+str(i)] = ineq_constraint
            
    return ineq_constraints_dict


########
#運動学モデルに基づく等式制約g(x)
########

#運動学モデルに基づく制約方程式3個と(N-1)の積だけ等式制約を生成。
def generate_model_eq_constraints():
    
    #model_eq_constraintsを保存する辞書の生成
    x_model_eq_constraints_dict = {}
    y_model_eq_constraints_dict = {}
    theta_model_eq_constraints_dict = {}
    
    #i番目と(i+1)番目のWayPointに関する3つの運動学モデルの制約を定義する
    for i in range(0, p.N - 1):
        
        #xに関する運動学モデルの制約
        def x_model_eq_constraint(trajectory_vector):
            
            trajectory_matrix = vector_to_matrix(trajectory_vector)
            
            return trajectory_matrix[0, i+1] - (trajectory_matrix[0, i] + p.v * np.cos(trajectory_matrix[2, i]) * p.dt)
        
        #yに関する運動学モデルの制約
        def y_model_eq_constraint(trajectory_vector):
            
            trajectory_matrix = vector_to_matrix(trajectory_vector)
            
            return trajectory_matrix[1, i+1] - (trajectory_matrix[1, i] + p.v * np.sin(trajectory_matrix[2, i]) * p.dt)
        
        #thetaに関する運動学モデルの制約
        def theta_model_eq_constraint(trajectory_vector):
            
            trajectory_matrix = vector_to_matrix(trajectory_vector)
            
            return trajectory_matrix[2, i+1] - (trajectory_matrix[2, i] + p.v * np.tan(trajectory_matrix[3, i]) * p.dt / p.L)
        
        #辞書に定義した関数を追加
        x_model_eq_constraints_dict['x_model_eq_constraints'+'_'+str(i)] = x_model_eq_constraint
        
        y_model_eq_constraints_dict['y_model_eq_constraints'+'_'+str(i)] = y_model_eq_constraint
        
        theta_model_eq_constraints_dict['theta_model_eq_constraints'+'_'+str(i)] = theta_model_eq_constraint
        
    return x_model_eq_constraints_dict, y_model_eq_constraints_dict, theta_model_eq_constraints_dict


########
#初期状態、終端状態における等式制約(境界条件)
########

#状態変数4個×2の境界条件を生成
def generate_boundary_condition():
    
    #境界条件を保存する辞書の作成
    boundary_condition_dict = {}
    
    ########初期状態に関する等式制約の関数定義
    
    #xの初期状態の制約
    def x_ini_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[0, 0] - p.initial_x
    
    #yの初期状態の制約
    def y_ini_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[1, 0] - p.initial_y
    
    #thetaの初期状態の制約
    def theta_ini_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[2, 0] - p.initial_theta

    #phiの初期状態の制約
    def phi_ini_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[3, 0] - p.initial_phi


    ########終端状態に関する等式制約の関数定義
    
    #xの終端状態の制約
    def x_ter_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[0, p.N - 1] - p.terminal_x
    
    #yの終端状態の制約
    def y_ter_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[1, p.N - 1] - p.terminal_y
    
    #xの終端状態の制約
    def theta_ter_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[2, p.N - 1] - p.terminal_theta
    
    #xの終端状態の制約
    def phi_ter_boundary_condition(trajectory_vector):
        
        trajectory_matrix = vector_to_matrix(trajectory_vector)
        
        return trajectory_matrix[0, p.N - 1] - p.terminal_phi
    
    #定義した関数を辞書に追加
    boundary_condition_dict['x_ini_boundary_condition'] = x_ini_boundary_condition
    
    boundary_condition_dict['y_ini_boundary_condition'] = y_ini_boundary_condition
    
    boundary_condition_dict['theta_ini_boundary_condition'] = theta_ini_boundary_condition
    
    boundary_condition_dict['phi_ini_boundary_condition'] = phi_ini_boundary_condition
    
    boundary_condition_dict['x_ter_boundary_condition'] = x_ter_boundary_condition
    
    boundary_condition_dict['y_ter_boundary_condition'] = y_ter_boundary_condition
    
    boundary_condition_dict['theta_ter_boundary_condition'] = theta_ter_boundary_condition
    
    boundary_condition_dict['phi_ter_boundary_condition'] = phi_ter_boundary_condition
    
    return boundary_condition_dict


########
#不等式制約と等式制約をまとめる関数
########

def generate_constraints():
    #不等式制約の辞書を関数から作成
    ineq_constraint_dict = generate_ineq_constraints()
    
    #modelの等式制約の辞書を関数から作成
    x_model_eq_constraints_dict, y_model_eq_constraints_dict, theta_model_eq_constraints_dict = generate_model_eq_constraints()
    
    #境界条件の辞書を関数から作成
    boundary_condition_dict = generate_boundary_condition()
    
    #全ての制約を含むリストを作成
    constraints = []
    
    #不等式制約を追加
    for g in ineq_constraint_dict.values():
        constraints.append({'type': 'ineq', 'fun': g})
    
    #modelの等式条件を追加
    #x
    for h in x_model_eq_constraints_dict.values():
        constraints.append({'type': 'eq', 'fun': h})
    #y
    for h in y_model_eq_constraints_dict.values():
        constraints.append({'type': 'eq', 'fun': h})
    #theta
    for h in theta_model_eq_constraints_dict.values():
        constraints.append({'type': 'eq', 'fun': h})
        
    #境界条件の等式制約
    for h in boundary_condition_dict.values():
        constraints.append({'type': 'eq', 'fun': h})
    
    return constraints


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
    