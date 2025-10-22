import numpy as np
import time
from queue import PriorityQueue

from functions import action_cost, heuristic_1, heuristic_octile, reach_goal

class Node:
    def __init__(self, x_in, y_in, theta_in, g_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.cost = g_in

    def print_node(self):
        print("\tNode :", "x =", self.x, "y =",self.y, "theta =", self.theta)

    def xytg(self):
        return self.x, self.y, self.theta, self.cost

    def __eq__(self, other):
        return (self.x, self.y, self.theta)==(other.x, other.y, other.theta)
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))



def Astar(start_config, goal_config, collision_fn, heuristic):
    start_x, start_y, start_theta = start_config
    goal_x, goal_y, goal_theta = goal_config

    neighbors_dir_4 = [(1,0), (0,1), (-1,0), (0,-1)]
    neighbors_dir_8 = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    angles = [0, np.pi/2, np.pi, -np.pi/2]

    # directions = neighbors_dir_4
    directions = neighbors_dir_8

    step_size = 0.125
    threshold = 0.1
    iter = 0
    success = False
    start_cost = 0

    path = []
    path_draw = []
    collision_free = []
    collided = []
    open_set = set()
    closed_set = set()
    parents = {}
    g_cost = {}

    def total_cost(prev_g, prev_x, prev_y, prev_theta, x, y, theta, gx, gy, gtheta):
        g_cost = prev_g + action_cost(prev_x, prev_y, prev_theta, x, y, theta)
        h = heuristic(x, y, theta, gx, gy, gtheta)
        # f = (0.4*g_cost + 0.6*h)
        f = g_cost + h
        return f

    q = PriorityQueue()
    queue_id = 0
    start_node = Node(start_x, start_y, start_theta, start_cost)
    q.put((total_cost(0, start_x, start_y, start_theta, start_x, start_y, start_theta, goal_x, goal_y, goal_theta), queue_id, start_node))

    open_set.add(start_node)
    parents[start_node] = None

    while not q.empty() and iter < 100000:
        iter += 1
        current_priority, current_id, current_node = q.get()
        current_x, current_y, current_theta, current_g_cost = current_node.xytg()

        # If the heuristic between current position and goal is smaller than threshold, we find the path
        # if(heuristic(current_x, current_y, current_theta, goal_x, goal_y, goal_theta) <= threshold):
        if reach_goal(current_x, current_y, current_theta, goal_x, goal_y, goal_theta):
            while current_node:
                path.append((current_node.x, current_node.y, current_node.theta))
                path_draw.append((current_node.x, current_node.y, 0.2))
                current_node = parents[current_node]
            path.reverse()
            success = True
            break

        closed_set.add(current_node)
        # open_set.remove(current_node)

        for dx, dy in directions:
            for angle in angles:
                queue_id += 1
                neighbor_x = current_x + dx * step_size
                neighbor_y = current_y + dy * step_size
                neighbor_theta = angle
                neighbor_g_cost = current_g_cost + action_cost(current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta)
                neighbor_node = Node(neighbor_x, neighbor_y, neighbor_theta, neighbor_g_cost)
                
                if neighbor_node in closed_set:
                    continue

                if collision_fn((neighbor_x, neighbor_y, neighbor_theta)):
                    collided.append((neighbor_x, neighbor_y, 0.1))
                    continue
                
                collision_free.append((neighbor_x, neighbor_y, 0.1))
                
                if (neighbor_node not in open_set) or (neighbor_g_cost < g_cost.get(neighbor_node, float('inf'))):
                    f_cost = total_cost(current_g_cost, current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta, goal_x, goal_y, goal_theta)
                    queue_id += 1
                    q.put((f_cost, queue_id, neighbor_node))
                    open_set.add(neighbor_node)
                    parents[neighbor_node] = current_node
                    g_cost[neighbor_node] = neighbor_g_cost

                    # check consistency
                    # if heuristic(current_x, current_y, current_theta, goal_x, goal_y, goal_theta) > action_cost(current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta) + heuristic(neighbor_x, neighbor_y, neighbor_theta, goal_x, goal_y, goal_theta):
                    #     print("this heuristic is not consistent")

    if success:
        total_path_cost = 0
        for i in range(len(path) - 1):
            x1, y1, theta1 = path[i]
            x2, y2, theta2 = path[i+1]
            total_path_cost += action_cost(x1, y1, theta1, x2, y2, theta2)
        print("Solution Found. Total Path Cost = ", total_path_cost)
    if not success:
        print("No Solution Found")

    return path, path_draw, collision_free, collided, total_path_cost
