import numpy as np
import perls2.controllers.utils.transform_utils as T
import scipy


class Model:

    def __init__(self, offset_mass_matrix=True):

        # robot states
        self.ee_pos = None
        self.ee_ori_quat = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tau = None
        self.joint_dim = None

        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None
        self.offset_mass_matrix = offset_mass_matrix
        self.mass_matrix_offset_val = [0.05, 0.05, 0.05]
        self.torque_compensation = None
        self.nullspace = None


    def update_states(self,
                      ee_pos,
                      ee_ori,
                      ee_pos_vel,
                      ee_ori_vel,
                      joint_pos,
                      joint_vel,
                      joint_tau,
                      joint_dim=None, 
                      torque_compensation=None):

        self.ee_pos = ee_pos

        if ee_ori.shape == (3, 3):
            self.ee_ori_mat = ee_ori
            self.ee_ori_quat = T.mat2quat(ee_ori)
        elif ee_ori.shape[0] == 4:
            self.ee_ori_quat = ee_ori
            self.ee_ori_mat = T.quat2mat(ee_ori)
        else:
            raise ValueError("orientation is not quaternion or matrix")

        self.ee_pos_vel = ee_pos_vel
        self.ee_ori_vel = ee_ori_vel

        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.joint_tau = joint_tau

        # Only update the joint_dim and torque_compensation attributes if it hasn't been updated in the past
        if not self.joint_dim:
            if joint_dim is not None:
                # User has specified explicit joint dimension
                self.joint_dim = joint_dim
            else:
                # Default to joint_pos length
                self.joint_dim = len(joint_pos)
            # Update torque_compensation accordingly
            #self.torque_compensation = np.zeros(self.joint_dim)
        self.torque_compensation = np.asarray(torque_compensation)

    def update_model(self,
                     J_pos,
                     J_ori,
                     mass_matrix):

        self.mass_matrix = mass_matrix
        if self.offset_mass_matrix:
          mm_weight_indices = [(4,4), (5,5), (6,6)] 
          for i in range(3):
            self.mass_matrix[mm_weight_indices[i]] += self.mass_matrix_offset_val[i]

        self.J_full = np.concatenate((J_pos, J_ori))
        self.J_pos = J_pos
        self.J_ori = J_ori

    def update(self):
      pass