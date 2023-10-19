"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np

import env
from RRT import Node


class Utils:
    def __init__(self):
        self.env = env.Env()
        
        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    #マージンを考慮した矩形障害物の4頂点をlistに保存
    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    #円形領域とセグメントとの衝突判定
    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        #セグメントの長さが0の時、衝突は発生しない
        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    #衝突判定を行う関数。衝突が発生するならTrue,そうでなければFalseを返す
    def is_collision(self, start, end):
        #セグメントの端点が障害物領域外にあるかどうか
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        #矩形障害物の4頂点とセグメントが交わるかどうかをチェック
        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        #円形領域とセグメントが交わるかどうかをチェック
        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    #ノードが障害物領域外にあればFalse,そうでなければTrueを返す
    def is_inside_obs(self, node):
        #ある程度マージンを確保するための変数delta
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False
    
    #pathのノードを減らす後処理
    def post_processing(self, path):
        length = len(path)
        processed_path = [path[0]]
        i, j = 0, 1
        while True:
            start = Node(path[i])
            end = Node(path[j])
            
            #衝突する場合
            if self.is_collision(start, end):
                processed_path.append(path[j-1])
                i = j - 1
                
            #衝突しない場合   
            else:
                j += 1
                if j == length:
                    break
                else:
                    continue
        processed_path.append(path[-1])
        
        return processed_path
                        
        
        
    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)

    
    ########
    #設計変数の行列(M×N)をベクトル(1×MN)に変換する関数
    ########
    def matrix_to_vector(trajectory_matrix):
        
        trajectory_vector = trajectory_matrix.flatten()
        
        return trajectory_vector

    ########
    #設計変数のベクトル(1×MN)を行列(M×N)をに変換する関数
    ########
    def vector_to_matrix(self, trajectory_vector, N, M):
        
        trajectory_matrix = trajectory_vector.reshape(M, N)
        
        return trajectory_matrix