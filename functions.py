import numpy as np

def action_cost(nx, ny, ntheta, mx, my, mtheta):
    cost = np.sqrt((nx - mx)**2 + (ny - my)**2 + min(abs(ntheta - mtheta), 2*np.pi - abs(ntheta - mtheta))**2)
    return cost

def heuristic_1(nx, ny, ntheta, gx, gy, gtheta):
    h = np.sqrt((nx - gx)**2 + (ny - gy)**2 + min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def heuristic_2(nx, ny, ntheta, gx, gy, gtheta):
    weight_theta = 1.5
    h = np.sqrt((nx - gx)**2 + (ny - gy)**2 + weight_theta * min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def heuristic_3(nx, ny, ntheta, gx, gy, gtheta):
    weight_theta = 0.6
    h = np.sqrt((nx - gx)**2 + (ny - gy)**2 + weight_theta * min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def heuristic_4(nx, ny, ntheta, gx, gy, gtheta):
    weight = 1.25
    h = weight * np.sqrt((nx - gx)**2 + (ny - gy)**2 + min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def heuristic_5(nx, ny, ntheta, gx, gy, gtheta):
    weight_trans = 1.5
    h = np.sqrt(weight_trans * ((nx - gx)**2 + (ny - gy)**2 )+ min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def heuristic_octile(nx, ny, ntheta, gx, gy, gtheta):
    dx = np.abs(nx - gx)
    dy = np.abs(ny - gy)
    h_trans = max(dx, dy) + (np.sqrt(2) - 1)*min(dx, dy)
    h_rot = min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))
    h = h_trans + h_rot
    return h

def heuristic_7(nx, ny, ntheta, gx, gy, gtheta):
    h = np.sqrt(((nx - gx)**2 + (ny - gy)**2 )) + 0.8 * min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))
    return h

def key_e(G_value, g_cost, h_cost):
    return (G_value - g_cost) / h_cost

def reach_goal(current_x, current_y, current_theta, goal_x, goal_y, goal_theta):
    h_thres = 0.125
    if heuristic_1(current_x, current_y, current_theta, goal_x, goal_y, goal_theta) < h_thres:
        return True
    return False
