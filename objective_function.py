from param import Parameter as p

def objective_function(x):
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += trajectory_matrix[3, i] ** 2
    
    return sum / p.N