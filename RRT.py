"""
RRT
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import env, plotting, utils

#ノードを作成するclass
class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

#RRTのメインclass
class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            #新しいセグメントが障害物領域外なら追加
            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                #追加したノードとゴールの距離を計算
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)
                
                #距離が閾値以下かつそのセグメントが領域外ならゴールの親ノードに新しいノードを設定し、pathの探索を終了
                if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    self.sampling_number = i
                    return self.extract_path(node_new)
           
        return None

    #ランダムにノードをサンプリングするclass
    def generate_random_node(self, goal_sample_rate):
        #deltaはいまのところ不明
        #おそらく壁際のサンプリングをしないように調整している
        delta = self.utils.delta
        #ある一定確率でゴールをサンプリングする
        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal
    
    #サンプリングしたノードに最も近いサンプリング済みのノードを返す
    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    #ランダムにサンプリングしたノードとそれに最も近いノードから新たに追加するノードを生成
    def new_state(self, node_start, node_end):
        #2点間のノードの距離と角度を計算
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        #距離と事前に設定した閾値の小さい方を距離として採用
        dist = min(self.step_len, dist)
        #新しいノードを生成
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        #追加したノードの親ノードを設定
        node_new.parent = node_start

        return node_new
    
    #親ノードをたどってpathを生成
    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

#RRTアルゴリズムを実行するmain関数
def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node
    
    #0.05の確率でゴールのノードをサンプリング
    rrt = Rrt(x_start, x_goal, 0.5, 0.05, 10000)
    path = rrt.planning()
    processed_path = rrt.utils.post_processing(path)
    print(processed_path)
    #アニメーションの作成
    if path:
        rrt.plotting.animation(rrt.vertex, path, "RRT", True)
        rrt.plotting.animation(rrt.vertex, processed_path, "RRT", False)
    else:
        print("No Path Found!")

        
if __name__ == '__main__':
    main()