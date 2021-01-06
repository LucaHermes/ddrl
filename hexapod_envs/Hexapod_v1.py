import numpy as np

import mujoco_py

from scipy import ndimage
from scipy.signal import convolve2d

import os

from math import sqrt, acos, fabs

from gym.envs.mujoco import mujoco_env
from gym import utils

#from gym.envs.mujoco.ant_v3 import AntEnv

DEFAULT_CAMERA_CONFIG = {
    'distance': 5.,
    'type': 1,
    'trackbodyid': 1,
    'elevation': -20.0,
}

def create_new_hfield(mj_model, smoothness = 0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters). 
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps))[0:mj_model.hfield_nrow[0],0:mj_model.hfield_ncol[0]]
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 8
    fromrow, torow = h_center - int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size,patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    print("CREATED RANDOM FIELD ", np.min(hfield), np.max(hfield), smoothness, bump_scale)
    mj_model.hfield_data[:] = hfield.ravel()

class HexapodEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    #MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/")
    
    def __init__(self,
                 xml_file='Hexapod_PhantomX_smallJointRanges.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.025, 1.5),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 frame_skip=5,
                 exclude_current_positions_from_observation=True,
                 hf_smoothness=1.):
        utils.EzPickle.__init__(**locals())
        
        #self.leg_list = ["coxa_fl_geom","coxa_fr_geom","coxa_rr_geom","coxa_rl_geom","coxa_mr_geom","coxa_ml_geom"]
        
        self.target_vel = np.array([0.24])
        
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        
        self.ctrl_cost_weight = self._ctrl_cost_weight
        self.contact_cost_weight = self._contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.modelpath = os.path.join(os.path.dirname(__file__), 'assets', xml_file)
        
        self.hf_smoothness = hf_smoothness
        # Scaled to 1. as six legged robot is smaller compared to Ant environment
        self.hf_bump_scale = 2.
        
        self.max_steps = 1000

#        self.joints_rads_low = np.array([-0.6, -1., -1.] * 6)
 #       self.joints_rads_high = np.array([0.6, 0.3, 1.] * 6)
  #      self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        # PID params
   #     self.Kp = 0.8
    #    self.Ki = 0
     #   self.Kd = 1
      #  self.int_err, self.past_err = 0, 0

        self.start_pos = None
        self.step_counter = 0
        self.ctrl_costs = 0.
        self.contact_costs = 0.
        self.vel_rewards = 0.
        self.upright_vector = np.array([0.,0.,1.])
        self.healthy_rewards = 0
        self.sum_rewards = 0
        
        mujoco_env.MujocoEnv.__init__(self, self.modelpath, frame_skip)
        #print("Mass: ", mujoco_py.functions.mj_getTotalmass(self.model))
        self.start_pos = self.sim.data.qpos[0].copy()
        
 #       self.model.nconmax = 1000 
  #      self.model.njmax = 2000

    def reset(self):
        obs = super().reset()
        # Move agent above heightfield
        h_center = int(0.5 * self.model.hfield_ncol)
        v_center = int(0.5 * self.model.hfield_nrow)
        initial_height = 0.2 + np.max(self.model.hfield_data.reshape(4000,400)[(h_center-3):(h_center+4),(v_center-3):(v_center+4)])
        old_pos = self.sim.data.qpos[2]
        self.sim.data.qpos[2] = initial_height
        #print("RESET FROM : " , old_pos, initial_height)
        return self._get_obs()

    def create_new_random_hfield(self):
        create_new_hfield(self.model, self.hf_smoothness, self.hf_bump_scale)
        self.reset()

    def set_hf_parameter(self, smoothness, bump_scale=None):
        self.hf_smoothness = smoothness
        if bump_scale:
            self.hf_bump_scale = bump_scale

    def set_target_velocity(self, t_vel):
        self.target_vel = np.array([t_vel])

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def healthy_reward(self):
        # Calculate if model keeps upright
        # Current orientation as a matrix
        torso_orient_mat = self.sim.data.body_xmat[1].reshape(3,3)
        # Reward is projection of z axis of body onto world z-axis
        healthy_reward = np.matmul(torso_orient_mat, self.upright_vector)[2]#0. #self.healthy_reward
        return (healthy_reward * self._healthy_reward)

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

#    def scale_action(self, action):
 #       return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def step(self, action): #setpoints):
        # From ant
        xy_position_before = self.get_body_com("torso")[:2].copy()
        
        # motor control using a PID controller
        ######################################
        # compute torques
#        joint_positions = self.sim.data.qpos.flat[-18:]
 #       joint_velocities = self.sim.data.qvel.flat[-18:]
        
        # limit motor maximum speed (this matches the real servo motors)
  #      timestep = self.dt
   #     vel_limit = 0.1  # rotational units/s
        #motor_setpoints = np.clip(2 * setpoints, joint_positions - timestep*vel_limit, joint_positions + timestep*vel_limit)

        # joint positions are scaled somehow roughly between -1.8...1.8
        # to meet these limits, multiply setpoints by two.
    #    err = 2 * setpoints - joint_positions
     #   self.int_err += err
      #  d_err = err - self.past_err
       # self.past_err = err
        
        #torques = np.minimum(
         #   1,
          #  np.maximum(-1, self.Kp * err + self.Ki * self.int_err + self.Kd * d_err),
#        )
        
        # clip available torque if the joint is moving too fast
 #       lowered_torque = 0.0
  #      torques = np.clip(torques,
   #         np.minimum(-lowered_torque, (-vel_limit-np.minimum(0, joint_velocities)) / vel_limit),
    #        np.maximum(lowered_torque, (vel_limit-np.maximum(0, joint_velocities)) / vel_limit))
        #print("Torques for joints: ", torques)
     #   self.do_simulation(torques, self.frame_skip)
        
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Reward calculation
        # use scaled action (see above)
        ctrl_cost = self.control_cost(action) #torques
        contact_cost = self.contact_cost

        #forward_reward = x_velocity #* 10 # Scaled as ant-sim env is much bigger
        forward_reward = (1. + 1./self.target_vel[0]) * (1. / (np.abs(x_velocity - self.target_vel[0]) + 1.) - 1. / (self.target_vel[0] + 1.))
        
        healthy_reward = self.healthy_reward
        
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        self.ctrl_costs += ctrl_cost
        self.contact_costs += contact_cost
        self.vel_rewards += forward_reward
        self.healthy_rewards += healthy_reward

        reward = rewards - costs
        self.sum_rewards += reward
        #if (self.step_counter % 50 == 0):
         #   print("REW: ", reward, forward_reward, healthy_reward)
        done = self.done
        
        if (healthy_reward < -0.8):
            done = True
            reward += (self.step_counter - self.max_steps)
        
        self.step_counter += 1
        
        if done or self.step_counter == self.max_steps:
            distance = (self.sim.data.qpos[0] - self.start_pos)# / (self.step_counter * self.dt)
            print("PhantomX target vel episode: ", distance, \
                (distance/ (self.step_counter * self.dt)), self.target_vel[0], \
                x_velocity, self.vel_rewards, self.sim.get_state().qvel.tolist()[0], \
                " / ctrl: ", self.ctrl_costs, self.ctrl_cost_weight, \
                " / contact: ", self.contact_costs, self.contact_cost_weight, \
                " / healthy: ", self.healthy_rewards, \
                " overall: ", self.sum_rewards, self.step_counter)
        
        observation = self._get_obs()
        
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        """ 
        Observation space for the Hexapod model.
        
        Following observation spaces are used: 
        * position information
        * velocity information
        * passive forces acting on the joints
        * last control signal
    
        
        For measured observations (basically everything starting with a q) ordering is:
            FL: 0, 1, 2
            FR: 3, 4, 5
            ML: 6, 7, 8
            MR: 9, 10, 11
            HL: 12, 13, 14
            HR: 15, 16, 17
            Important: plus offset! The first entries are global coordinates, velocities.
        """
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        #contact_force = self.contact_forces.flat.copy()
        # Provide passive force instead -- in joint reference frame = eight dimensions
        # joint_passive_forces = self.sim.data.qfrc_passive.flat.copy()[6:]
        # Sensor measurements in the joint:
        # qfrc_unc is the sum of all forces outside constraints (passive, actuation, gravity, applied etc)
        # qfrc_constraint is the sum of all constraint forces. 
        # If you add up these two quantities you get the total force acting on each joint
        # which is what a torque sensor should measure.
        # See note in http://www.mujoco.org/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181/
        joint_sensor_forces = self.sim.data.qfrc_unc[6:] + self.sim.data.qfrc_constraint[6:]

        # Provide actions from last time step (as used in the simulator = clipped)
        last_control = self.sim.data.ctrl.flat.copy()
        
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, joint_sensor_forces, last_control, self.target_vel))#, last_control)) #, contact_force))

        return observations


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        self.int_err = 0
        self.past_err = 0

        self.start_pos = self.sim.data.qpos[0] #self.get_body_com("torso")[:2].copy()
        self.step_counter = 0
        self.ctrl_costs = 0.
        self.contact_costs = 0.
        self.vel_rewards = 0.
        self.healthy_rewards = 0.
        self.sum_rewards = 0.
        
        return observation
     
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)