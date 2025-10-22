import numpy as np
from functions import action_cost, reach_goal, key_e
from queue import PriorityQueue
import time

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


def ANAstar(start_config, goal_config, collision_fn, heuristic):
    start_x, start_y, start_theta = start_config
    goal_x, goal_y, goal_theta = goal_config

    start_g = 0
    start_node = Node(start_x, start_y, start_theta, start_g)
    start_time = time.time()

    neighbors_dir_4 = [(1,0), (0,1), (-1,0), (0,-1)]
    neighbors_dir_8 = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    angles = [0, np.pi/2, np.pi, -np.pi/2]

    # directions = neighbors_dir_4
    directions = neighbors_dir_8

    step_size = 0.125
    success = False

    path = []
    path_draw = []
    collision_free = []
    collided = []
    open_set = set()
    closed_set = set()
    parents = {}
    open_set.add(start_node)
    parents[start_node] = None
    g_cost = {start_node:0}
    collide_node = []

    ana_G_cost = []
    ana_cost_time_stamp = []
    ana_E = []
    ana_E_time_stamp = []

    G_best = 1000000
    E = 1000000
    OPEN = PriorityQueue()
    start_h = heuristic(start_x, start_y, start_theta, goal_x, goal_y, goal_theta)
    start_e = (G_best / start_h)
    OPEN.put((1/start_e, 0, start_node))
    e_value = {start_node:start_e}

    node_id = 1

    def Improve_Solution(G, E, node_id, path, path_draw, OPEN):
        success = False
        iter_is = 0
        # iter_is_max = 1000
        while not OPEN.empty():
            closed_set.clear()
            current_priority, current_id, current_node = OPEN.get()
            current_x, current_y, current_theta, current_g_cost = current_node.xytg()
            g_cost[current_node] = current_g_cost
            e_s = current_priority

            if 1/e_s < E:
                E = 1/e_s
                if E < 100:
                    ana_E.append(E)
                    ana_E_time_stamp.append(time.time() - start_time)
            
            # check if reach the goal
            if reach_goal(current_x, current_y, current_theta, goal_x, goal_y, goal_theta):
                path = []
                path_draw = []
                while current_node:
                    curr_x, curr_y, curr_theta, curr_g = current_node.xytg()
                    path.append((curr_x, curr_y, curr_theta))
                    path_draw.append((curr_x, curr_y, 0.2))
                    current_node = parents[current_node]
                path.reverse()
                success = True
                G = 0
                for i in range(len(path) - 1):
                    x1, y1, theta1 = path[i]
                    x2, y2, theta2 = path[i+1]
                    G += action_cost(x1, y1, theta1, x2, y2, theta2)
                break

            closed_set.add(current_node)

            # For each successor s' of s do
            for dx, dy in directions:
                for angle in angles:
                    neighbor_x = current_x + dx * step_size
                    neighbor_y = current_y + dy * step_size
                    neighbor_theta = angle

                    dist_current_neighbor = action_cost(current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta)
                    neighbor_g_cost = current_g_cost + dist_current_neighbor
                    neighbor_node = Node(neighbor_x, neighbor_y, neighbor_theta, neighbor_g_cost)
                    
                    if neighbor_node in closed_set:
                        continue

                    if neighbor_node in collide_node:
                        continue
                    elif collision_fn((neighbor_x, neighbor_y, neighbor_theta)):
                        collide_node.append(neighbor_node)
                        continue
       

                    # if g(s) + c(s,s') < g(s') then
                    if neighbor_node not in g_cost or g_cost[current_node] + dist_current_neighbor < g_cost[neighbor_node]:
                        #  g(s') <- g(s) + c(s,s')
                        g_cost[neighbor_node] = g_cost[current_node] + dist_current_neighbor
                        # pred(s') <- s
                        parents[neighbor_node] = current_node

                        # if g(s') + h(s') < G then insert or update s' in OPEN with key e(s')
                        h_neighbor = heuristic(neighbor_x, neighbor_y, neighbor_theta, goal_x, goal_y, goal_theta)
                        if g_cost[neighbor_node] + h_neighbor >= G:
                            continue
                        
                        # e_neighbor = (G - g_cost[neighbor_node]) / h_neighbor
                        e_neighbor = key_e(G, g_cost[neighbor_node], h_neighbor)
                        # e_score[neighbor_node] = e_neighbor

                        if e_neighbor < 1:
                            continue
                        
                        if g_cost[neighbor_node] + h_neighbor < G:
                            if neighbor_node in open_set:
                                if (1/e_neighbor) < (1/e_value[neighbor_node]):
                                    new_open = PriorityQueue()
                                    old_open = OPEN
                                    while not old_open.empty():
                                        temp_priority, temp_id, temp_node = old_open.get()
                                        if temp_node == neighbor_node:
                                            new_open.put((1/e_neighbor, temp_id, neighbor_node))
                                            e_value[neighbor_node] = e_neighbor
                                        else:
                                            new_open.put((temp_priority, temp_id, temp_node))
                                    OPEN = new_open
                                    node_id += 1 
                            else:
                                open_set.add(neighbor_node)
                                OPEN.put((1/e_neighbor, node_id, neighbor_node))
                                e_value[neighbor_node] = e_neighbor
                                node_id += 1
                                
        return path, path_draw, G, E, OPEN


    while not OPEN.empty():
        path, path_draw, G_new, E_new, OPEN = Improve_Solution(G_best, E, node_id, path, path_draw, OPEN)
        E = E_new

        ana_G_cost.append(G_best)
        ana_cost_time_stamp.append(time.time() - start_time)

        if G_new != G_best:
            # ana_G_cost.append(G_best)
            # ana_cost_time_stamp.append(time.time() - start_time)
            G_best = G_new
            print("G update")
            print(f"Current G: {G_best}")
            print(f"Current E: {E}")
            print(f"Current runtime: {time.time() - start_time}")
            ana_G_cost.append(G_best)
            ana_cost_time_stamp.append(time.time() - start_time)
        else:
            # ana_G_cost.append(G_best)
            # ana_cost_time_stamp.append(time.time() - start_time)
            break

        new_Open = PriorityQueue()
        open_set.clear()

        # update keys e(s) in OPEN and prune if g(s) + h(s) >= G
        while not OPEN.empty():
            temp_priority, temp_id, temp_node = OPEN.get()
            temp_x, temp_y, temp_theta, temp_g = temp_node.xytg()
            h_temp = heuristic(temp_x, temp_y, temp_theta, goal_x, goal_y, goal_theta)
            e_value[temp_node] = key_e(G_best, g_cost[temp_node], h_temp)
            if g_cost[temp_node] + h_temp < G_best:
                new_Open.put((1/e_value[temp_node], temp_id, temp_node))
                open_set.add(temp_node)
        OPEN = new_Open

    return path, path_draw, collision_free, collided, ana_G_cost, ana_cost_time_stamp, ana_E, ana_E_time_stamp