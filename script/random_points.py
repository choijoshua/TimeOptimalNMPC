import numpy as np
import csv

import copy

import sys, os
BASEPATH = os.path.abspath(__file__).split("script/", 1)[0]
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel
from nmpc import Nmpc
from nmpc_params import NmpcParams

max_time = 0.8
number_of_states = 1500


class Optimization():
    def __init__(self, quad:QuadrotorModel, params:NmpcParams):
        self._quad = quad
        self._params = params

    def solve(self, xinit, xend):
        nmpc = Nmpc(quad, params, xinit, xend)

        dts = np.array([0.3]*(params._wpt_num + 1))

        nmpc.define_opt()
        res = nmpc.solve_opt(dts)

        nmpc.define_opt_t()
        res_t = nmpc.solve_opt_t()
        # Save the trajectory inside the results folder
        return self.minimum_time(res_t, nmpc) 


    def minimum_time(self, res, ctr: Nmpc):
        x = res['x'].full().flatten()
            
        t = 0
        s = ctr._xinit
        u = x[ctr._Horizon*ctr._X_dim: ctr._Horizon*ctr._X_dim+ctr._U_dim]
        initial_u = copy.deepcopy(u)
        u_last = u

        ###
        for i in range(ctr._seg_num):
            # ctrimized time gap
            dt = x[-(ctr._seg_num)+i]
            for j in range(ctr._Ns[i]):
                idx = ctr._N_wp_base[i]+j
                t += dt
                s = x[idx*ctr._X_dim: (idx+1)*ctr._X_dim]
                if idx != ctr._Horizon-1:
                    u = x[ctr._Horizon*ctr._X_dim+(idx+1)*ctr._U_dim: ctr._Horizon*ctr._X_dim+(idx+2)*ctr._U_dim]
                    u_last = u
                else:
                    u = u_last

        return t, initial_u, u

def generate_random_states(initial):

    if initial:
        pos_x = 0
        pos_y = 0
        pos_z = 0
    
    else:
        pos_x = 2.5 * np.random.uniform(-1, 1)
        pos_y = 2.5 * np.random.uniform(-1, 1)
        pos_z = 2.5 * np.random.uniform(-1, 1)

        while pos_x == 0 and pos_y == 0 and pos_z == 0:
            pos_x = 2.5 * np.random.uniform(-1, 1)
            pos_y = 2.5 * np.random.uniform(-1, 1)
            pos_z = 2.5 * np.random.uniform(-1, 1)


    vel_x = 5 * np.random.uniform(-1, 1)
    vel_y = 5 * np.random.uniform(-1, 1)
    vel_z = 5 * np.random.uniform(-1, 1)

    quaternion_w = np.random.rand()
    quaternion_x = np.random.rand()
    quaternion_y = np.random.rand()
    quaternion_z = np.random.rand()

    z = [quaternion_w, quaternion_x, quaternion_y, quaternion_z]
    z = z / np.linalg.norm(z)

    body_rate_x = 0
    body_rate_y = 0
    body_rate_z = 0

    return np.array([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, z[0], z[1], z[2], z[3], body_rate_x, body_rate_y, body_rate_z])

if __name__== "__main__":

    quad = QuadrotorModel(BASEPATH+'/parameters/quad_params.yaml')
    # Load NMPC optimization parameters
    params = NmpcParams(BASEPATH+'/parameters/ctr_params.yaml')
    # Instantiate an NMPC planner
    optimization = Optimization(quad, params)

    count = 0

    with open(BASEPATH+"/results/random_states.csv", 'a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        labels = ['t',
                "p_x", "p_y", "p_z",
                "v_x", "v_y", "v_z",
                "q_w", "q_x", "q_y", "q_z",
                "w_x", "w_y", "w_z",
                "a_lin_x", "a_lin_y", "a_lin_z",
                "a_rot_x", "a_rot_y", "a_rot_z",
                "u_1", "u_2", "u_3", "u_4",
                "jerk_x", "jerk_y", "jerk_z",
                "snap_x", "snap_y", "snap_z"]
        a_lin = [0,0,0]
        a_rot = [0,0,0]
        jerk = [0,0,0]
        snap = [0,0,0]

        while (count < number_of_states):

            # Set the initial state [position, velocity, quaternion (rotation), bodyrate]
            xinit = generate_random_states(True)
            # Set the target state [position, velocity, quaternion (rotation), bodyrate]
            xend = generate_random_states(False)
            
            time, init_u, u = optimization.solve(xinit, xend)
            print("--------------------")
            print(time)
            print(count)
            print("--------------------")
            if time < max_time:
                writer.writerow([0, xinit[0], xinit[1], xinit[2], xinit[3], xinit[4], xinit[5], xinit[6], xinit[7], xinit[8], xinit[9], xinit[10], xinit[11], xinit[12], a_lin[0], a_lin[1], a_lin[2], a_rot[0], a_rot[1], a_rot[2], init_u[0], init_u[1], init_u[2], init_u[3], jerk[0], jerk[1], jerk[2], snap[0], snap[1], snap[2]])
                writer.writerow([time, xend[0], xend[1], xend[2], xend[3], xend[4], xend[5], xend[6], xend[7], xend[8], xend[9], xend[10], xend[11], xend[12], a_lin[0], a_lin[1], a_lin[2], a_rot[0], a_rot[1], a_rot[2], u[0], u[1], u[2], u[3], jerk[0], jerk[1], jerk[2], snap[0], snap[1], snap[2]])
                count += 1








