import matplotlib.pyplot as plt
import matplotlib.patches as patches
from param import Parameter as p
import util
import numpy as np
import GenerateInitialPath

########
#壁と障害物の配置し表示する関数
########
def vis_env():
    fig, ax = plt.subplots()
    
    #wallを配置
    #左側
    leftside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(leftside_wall)
    #右側
    rightside_wall = patches.Rectangle((p.x_max, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(rightside_wall)
    #下側
    downside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min - p.wall_thick), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(downside_wall)
    #上側
    upside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_max), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(upside_wall)
    
    #障害物を配置
    for k in range(p.obstacle_list):
        x_o, y_o, r_o = p.obstacle_list[k][0], p.obstacle_list[k][1], p.obstacle_list[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='black')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    plt.show()
    
    return None
    
    
########    
#経路を環境に表示する関数
########
def vis_path(trajectory_vector):
    fig, ax = plt.subplots()
    
    #vectorをmatrixに変換
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    x, y = trajectory_matrix[0], trajectory_matrix[1]
    
    ax.scatter(x, y, marker='x', color='red', s=5)
    
    #wallを配置
    #左側
    leftside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(leftside_wall)
    #右側
    rightside_wall = patches.Rectangle((p.x_max, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(rightside_wall)
    #下側
    downside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min - p.wall_thick), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(downside_wall)
    #上側
    upside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_max), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(upside_wall)
    
    #障害物を配置
    for k in range(len(p.obstacle_list)):
        x_o, y_o, r_o = p.obstacle_list[k][0], p.obstacle_list[k][1], p.obstacle_list[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='black')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    
    plt.show()
    
    return None

########
#2本のpathを比較する関数
########
def compare_path(trajectory_vector1, trajectory_vector2):
    fig, ax = plt.subplots()
    
    #2本のpathを配置
    trajectory_matrix1 = util.vector_to_matrix(trajectory_vector1)
    x1, y1 = trajectory_matrix1[0], trajectory_matrix1[1]
    ax.scatter(x1, y1, marker='x', color='red', s=5, label='Initial path')
    
    trajectory_matrix2 = util.vector_to_matrix(trajectory_vector2)
    x2, y2 = trajectory_matrix2[0], trajectory_matrix2[1]
    ax.scatter(x2, y2, marker='x', color='blue', s=5, label='Optimized path')
    
    #wallを配置
    #左側
    leftside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(leftside_wall)
    #右側
    rightside_wall = patches.Rectangle((p.x_max, p.y_min), p.wall_thick, p.y_max - p.y_min, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(rightside_wall)
    #下側
    downside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_min - p.wall_thick), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(downside_wall)
    #上側
    upside_wall = patches.Rectangle((p.x_min - p.wall_thick, p.y_max), 2 * p.wall_thick + p.x_max - p.x_min, p.wall_thick, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(upside_wall)
    
    #障害物を配置
    for k in range(len(p.obstacle_list)):
        x_o, y_o, r_o = p.obstacle_list[k][0], p.obstacle_list[k][1], p.obstacle_list[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='black')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    ax.set_aspect('equal')
    ax.legend(loc="best")
    plt.show()
    
    return None
    
########
#姿勢thetaのグラフを生成
########
def vis_history_theta(trajectory_vector, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    
    theta = trajectory_matrix[2]
    
    ax.plot(theta, color='blue', label='theta[rad]')
    ax.set_xlabel('t')
    ax.set_ylabel('theta[rad]')
    ax.legend(loc='best')
    
    #thetaの範囲を追加
    if range_flag:
        theta_max_list = [p.theta_max for i in range(p.N)]
        theta_min_list = [p.theta_min for i in range(p.N)]
        ax.plot(theta_max_list, color='red', linestyle='-.')
        ax.plot(theta_min_list, color='red', linestyle='-.')
    else:
        pass
    
    plt.show()
    
    
########
#ステアリング角phiのグラフを生成
########
def vis_history_phi(trajectory_vector, range_flag = False):
    fig, ax = plt.subplots()
    
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    
    phi = trajectory_matrix[3]
    
    ax.plot(phi, color='blue', label='phi[rad]')
    ax.set_xlabel('t')
    ax.set_ylabel('phi[rad]')
    ax.legend(loc='best')
    
    #thetaの範囲を追加
    if range_flag:
        phi_max_list = [p.phi_max for i in range(p.N)]
        phi_min_list = [p.phi_min for i in range(p.N)]
        ax.plot(phi_max_list, color='red', linestyle='-.')
        ax.plot(phi_min_list, color='red', linestyle='-.')
    else:
        pass
    
    plt.show()
    
    