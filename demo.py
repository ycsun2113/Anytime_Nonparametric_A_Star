import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from queue import PriorityQueue
import matplotlib.pyplot as plt

from a_star import Astar
from ana_star import ANAstar
from functions import action_cost, heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5, heuristic_octile, heuristic_7

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

#########################

def main(screenshot=False):
    expected_runtime = ['1200 to 1700 seconds',  '300 to 700 seconds',  '300 to 700 seconds',  '1200 to 1700 seconds']
    print()
    print(f"Environment 0 Expected Runtime: {expected_runtime[0]}")
    print(f"Environment 1 Expected Runtime: {expected_runtime[1]}")
    print(f"Environment 2 Expected Runtime: {expected_runtime[2]}")
    print(f"Environment 3 Expected Runtime: {expected_runtime[3]}")

    env_number = int(input("Which environment you would like to run? (input 0~3): "))
    while env_number!=0 and env_number!=1 and env_number!=2 and env_number!=3 :
        env_number = int(input("Please input a number from 0 to 3: "))

    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    environments = ['pr2doorway.json', 'env_3.json', 'env_4.json', 'env_5.json']
    robots, obstacles = load_env(environments[env_number])
    # robots, obstacles = load_env('pr2doorway.json')

    print()
    print(f"Expected total runtime: {expected_runtime[env_number]}")

    cost_lower_limit = [10.5, 5.0, 5.3, 6.75]
    cost_upper_limit = [15, 5.3, 9, 7.05]

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))

    goals_config = [(2.6, -1.3, -np.pi/2), (2, -1.3, -np.pi/2), (1, -1, -np.pi/2), (2, -1.3, -np.pi/2)]
    # goal_config = (2.6, -1.3, -np.pi/2)
    goal_config = goals_config[env_number]
    # path = []
    start_time = time.time()

    ####################
    ### MY CODE HERE ###
    ####################
    sphere_radius = 0.05
    draw_configurations = False
    plot_results = True

    # Set the start config and draw the start point in the world:
    start_x, start_y, start_theta = start_config
    start_position = (start_x, start_y, 0.2)
    draw_sphere_marker(start_position, sphere_radius*2, (1, 0, 0, 1))

    # Set the goal config and draw the goal point in the world:
    goal_x, goal_y, goal_theta = goal_config
    goal_position = (goal_x, goal_y, 0.2)
    draw_sphere_marker(goal_position, sphere_radius*2, (0, 1, 0, 1))


    ##########################
    ### RUN ANA* Algorithm ###
    ##########################
    
    print("=======================================================")
    print("Running ANA* with heuristic_1 (euclidean)...")
    print()
    start_time_1 = time.time()
    path, path_draw, collision_free, collided, ana_G1, ana_Gt1, ana_E1, ana_Et1 = ANAstar(start_config, goal_config, collision_fn, heuristic_1)
    # path, path_draw, collision_free, collided, ana_G1, ana_Gt1, ana_E1, ana_Et1 = ANAstar(start_config, goal_config, collision_fn, heuristic_4)
    end_time_1 = time.time() - start_time_1
    print(f"\nANA* euclidean end \nruntime = {end_time_1}, cost = {ana_G1[-1]}")

    # print("=======================================================")
    # print("Running ANA* with heuristic_2...")
    # print()
    # start_time_2 = time.time()
    # path_2, path_draw_2, collision_free_2, collided_2, ana_G2, ana_Gt2, ana_E2, ana_Et2 = ANAstar(start_config, goal_config, collision_fn, heuristic_2)
    # end_time_2 = time.time() - start_time_2
    # print(f"\nANA* euclidean with rot_weight = 1.5 end \nruntime = {end_time_2}, cost = {ana_G2[-1]}")

    print("=======================================================")
    print("Running ANA* with heuristic_2 (w_theta = 0.6)...")
    print()
    start_time_3 = time.time()
    path_3, path_draw_3, collision_free_3, collided_3, ana_G3, ana_Gt3, ana_E3, ana_Et3 = ANAstar(start_config, goal_config, collision_fn, heuristic_3)
    end_time_3 = time.time() - start_time_3
    print(f"\nANA* euclidean with w_theta = 0.6 end \nruntime = {end_time_3}, cost = {ana_G3[-1]}")


    # print("=======================================================")
    # print("Running ANA* with heuristic_6 (octile)...")
    # print()
    # start_time_6 = time.time()
    # path_6, path_draw_6, collision_free_6, collided_6, ana_G6, ana_Gt6, ana_E6, ana_Et6 = ANAstar(start_config, goal_config, collision_fn, heuristic_octile)
    # end_time_6 = time.time() - start_time_6
    # print(f"\nANA* octile end \nruntime = {end_time_6}, cost = {ana_G6[-1]}")

    # print("=======================================================")
    # print("Running ANA* with heuristic_7...")
    # print()
    # start_time_7 = time.time()
    # path_7, path_draw_7, collision_free_7, collided_7, ana_G7, ana_Gt7, ana_E7, ana_Et7 = ANAstar(start_config, goal_config, collision_fn, heuristic_7)
    # end_time_7 = time.time() - start_time_7
    # print(f"\nANA* euclidean with rotation term outside end \nruntime = {end_time_7}, cost = {ana_G7[-1]}")
    
    
    ########################
    ### RUN A* Algorithm ###
    ########################

    print("=======================================================")
    print("Running A* with heuristic_1 (euclidean)...")
    print()
    start_time_astar = time.time()
    path_astar1, path_draw_astar1, collision_free_astar1, collided_astar1, astar_path_cost = Astar(start_config, goal_config, collision_fn, heuristic_1)
    end_time_astar = time.time() - start_time_astar
    astar_t1 = [0, end_time_astar, time.time() - start_time_astar]
    astar_cost_1 = [100, 100, astar_path_cost]
    print(f"\nA* euclidean end \nruntime = {end_time_astar}, cost = {astar_path_cost}")


    # print("=======================================================")
    # print("Running A* with heuristic_6 (octile)...")
    # print()
    # start_time_astar_3 = time.time()
    # path_astar3, path_draw_astar3, collision_free_astar3, collided_astar3, astar_path_cost_3 = Astar(start_config, goal_config, collision_fn, heuristic_octile)
    # end_time_astar_3 = time.time() - start_time_astar_3
    # astar_t3 = [0, end_time_astar_3, time.time() - start_time_astar_3]
    # astar_cost_3 = [100, 100, astar_path_cost_3]
    # print(f"\nA* octile end \nruntime = {end_time_astar_3}, cost = {astar_path_cost_3}")

    # print("=======================================================")
    # print("Running A* with heuristic_2 (w_theta = 1.5)...")
    # print()
    # start_time_astar_4 = time.time()
    # path_astar4, path_draw_astar4, collision_free_astar4, collided_astar4, astar_path_cost_4 = Astar(start_config, goal_config, collision_fn, heuristic_2)
    # end_time_astar_4 = time.time() - start_time_astar_4
    # astar_t4 = [0, end_time_astar_4, time.time() - start_time_astar_4]
    # astar_cost_4 = [100, 100, astar_path_cost_4]
    # print(f"\nA* euclidean with w_theta = 1.5 end \nruntime = {end_time_astar_4}, cost = {astar_path_cost_4}")

    print("=======================================================")
    print("Running A* with heuristic_2 (w_theta = 0.6)...")
    print()
    start_time_astar_5 = time.time()
    path_astar5, path_draw_astar5, collision_free_astar5, collided_astar5, astar_path_cost_5 = Astar(start_config, goal_config, collision_fn, heuristic_3)
    end_time_astar_5 = time.time() - start_time_astar_5
    astar_t5 = [0, end_time_astar_5, time.time() - start_time_astar_5]
    astar_cost_5 = [100, 100, astar_path_cost_5]
    print(f"\nA* euclidean with w_theta = 0.6 end \nruntime = {end_time_astar_5}, cost = {astar_path_cost_5}")

    # print("=======================================================")
    # print("Running A* with heuristic_7 (rot out)...")
    # print()
    # start_time_astar_7 = time.time()
    # path_astar7, path_draw_astar7, collision_free_astar7, collided_astar7, astar_path_cost_7 = Astar(start_config, goal_config, collision_fn, heuristic_7)
    # end_time_astar_7 = time.time() - start_time_astar_7
    # astar_t7 = [0, end_time_astar_7, time.time() - start_time_astar_7]
    # astar_cost_7 = [100, 100, astar_path_cost_7]
    # print(f"\nA* euclidean with rot out end \nruntime = {end_time_astar_7}, cost = {astar_path_cost_7}")


    #################
    ### DRAW PATH ###
    #################
    print("=======================================================")
    print("Drawing optimal paths...")
    print("Green path: ANA* (Euclidean)")
    # print("Red path: ANA* (CUstom h2)")
    print("Black path: A* (Euclidean)")
    # print("Blue path: A* (Custom h2)")
    for path_points in path_draw:
       draw_sphere_marker(path_points, sphere_radius, (0, 1, 0, 1)) 
    # for path_points in path_draw_3:
    #     draw_sphere_marker(path_points, sphere_radius, (1, 0, 0, 1)) 
    for path_points in path_draw_astar1:
        draw_sphere_marker(path_points, sphere_radius, (0, 0, 0, 1)) 
    # for path_points in path_draw_astar5:
    #     draw_sphere_marker(path_points, sphere_radius, (0, 0, 1, 1)) 


    print("=======================================================")
    print("Total runtime: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)


    ##############################
    ### PRINT RUNTIME AND COST ###
    ##############################
    print("=======================================================")
    print("Results:")
    print(f"ANA* euclidean, runtime = {end_time_1}, cost = {ana_G1[-1]}")
    # print(f"ANA* euclidean with weight_theta = 1.5, runtime = {end_time_2}, cost = {ana_G2[-1]}")
    print(f"ANA* euclidean with weight_theta = 0.6, runtime = {end_time_3}, cost = {ana_G3[-1]}")
    # print(f"ANA* euclidean with weight = 1.25, runtime = {end_time_4}, cost = {ana_G4[-1]}")
    # print(f"ANA* euclidean with weight_trans = 1.25, runtime = {end_time_5}, cost = {ana_G5[-1]}")
    # print(f"ANA* octile, runtime = {end_time_6}, cost = {ana_G6[-1]}")
    # print(f"ANA* euclidean with rot out, runtime = {end_time_7}, cost = {ana_G7[-1]}")

    print(f"A* euclidean, runtime = {end_time_astar}, cost = {astar_path_cost}")
    # print(f"A* euclidean with weight = 1.25, runtime = {end_time_astar_2}, cost = {astar_path_cost_2}")
    # print(f"A* octile, runtime = {end_time_astar_3}, cost = {astar_path_cost_3}")
    # print(f"A* euclidean with weight_theta = 1.5, runtime = {end_time_astar_4}, cost = {astar_path_cost_4}")
    print(f"A* euclidean with weight_theta = 0.6, runtime = {end_time_astar_5}, cost = {astar_path_cost_5}")
    # print(f"A* euclidean with weight_trans = 1.25, runtime = {end_time_astar_6}, cost = {astar_path_cost_6}")
    # print(f"A* euclidean with rot out, runtime = {end_time_astar_7}, cost = {astar_path_cost_7}")
    print("=======================================================")
    


    ####################
    ### PLOT RESULTS ###
    ####################
    if plot_results:
        print("Plotting...")
        plt.figure(1)
        plt.plot(ana_Et1, ana_E1, label="ANA* (euclidean)")
        # plt.plot(ana_Et2, ana_E2, label="ANA* (wθ = 1.5)")
        plt.plot(ana_Et3, ana_E3, label="ANA* (wθ = 0.6)")
        # plt.plot(ana_Et4, ana_E4, label="ANA* (w = 1.25)")
        # plt.plot(ana_Et5, ana_E5, label="ANA* (wt = 1.25)")
        # plt.plot(ana_Et6, ana_E6, label="ANA* (octile)")
        # plt.plot(ana_Et7, ana_E7, label="ANA* (rot out)")
        plt.xlabel("Time (sec)")
        plt.ylabel("Suboptimality")
        
        plt.legend(loc = 'upper right')
        
        plt.figure(2)
        plt.plot(ana_Gt1, ana_G1, label="ANA* (euclidean)")
        # plt.plot(ana_Gt2, ana_G2, label="ANA* (wθ = 1.5)")
        plt.plot(ana_Gt3, ana_G3, label="ANA* (wθ = 0.6)")
        # plt.plot(ana_Gt4, ana_G4, label="ANA* (w = 1.25)")
        # plt.plot(ana_Gt5, ana_G5, label="ANA* (wt = 1.25)")
        # plt.plot(ana_Gt6, ana_G6, label="ANA* (octile)")
        # plt.plot(ana_Gt7, ana_G7, label="ANA* (rot out)")

        plt.plot(astar_t1, astar_cost_1, label="A* (euclidean)")
        # plt.plot(astar_t2, astar_cost_2, label="A* (w = 1.25)")
        # plt.plot(astar_t4, astar_cost_4, label="A* (wθ = 1.5)")
        plt.plot(astar_t5, astar_cost_5, label="A* (wθ = 0.6)")
        # plt.plot(astar_t3, astar_cost_3, label="A* (octile)")
        # plt.plot(astar_t6, astar_cost_6, label="A* (wt = 1.25)")
        # plt.plot(astar_t7, astar_cost_7, label="A* (rot out)")
        plt.xlabel("Time (sec)")
        plt.ylabel("Solution Cost")
        plt.ylim(cost_lower_limit[env_number], cost_upper_limit[env_number])
        plt.legend(loc = 'upper right')

        plt.show()


    ######################
    
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
