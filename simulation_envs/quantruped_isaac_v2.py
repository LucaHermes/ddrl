from isaacgym import gymapi, gymtorch, gymutil
from gym import Env
from functools import partial, lru_cache#, cached_property
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
import time
# Sources:
# [1] https://github.com/openai/gym/blob/da7b8ae8fc6f5ff97c5f7303c03a26c8b4b3f4e2/gym/envs/mujoco/ant_v3.py#L230

def default_sim_params(gpu_enabled=False):
    sim_params = gymapi.SimParams()
    sim_params.physx.use_gpu = gpu_enabled
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.dt = 0.01
    return sim_params

ISAAC_CONFIG = {
    'num_envs' : 10,
    'sim_params' : default_sim_params(gpu_enabled=False),
    'headless' : True,
    'sim_device' : -1,
    'rendering_device' : -1,
    'sim_type' : gymapi.SIM_PHYSX
}


class IsaacEnv(Env):

    def __init__(self, num_envs=10, sim_params=None, headless=True, sim_device=None, rendering_device=None, sim_type=None):
        self.gym = gymapi.acquire_gym()
        self.sim = self.__create_sim(sim_params, sim_device, rendering_device, sim_type, headless)
        self.viewer = self.__create_viewer(headless)
        self._num_envs = num_envs
        self.camera_handles = dict()
        self.skip_frames = 5

    def __create_viewer(self, headless):
        if headless is False:
            cam_props = gymapi.CameraProperties()
            return self.gym.create_viewer(self.sim, cam_props)

    def __create_sim(self, sim_params, sim_device, rendering_device, sim_type, headless):
        if sim_type is None:
            sim_type = ISAAC_CONFIG['sim_type']
        if sim_params is None:
            #sim_params = gymapi.SimParams()
            sim_params = ISAAC_CONFIG['sim_params']
        if sim_device is None:
            sim_device = ISAAC_CONFIG['sim_device']
        if rendering_device is None:
            rendering_device = ISAAC_CONFIG['rendering_device']
        #if headless:
        #    rendering_device = -1
        sim = self.gym.create_sim(sim_device, rendering_device, sim_type, sim_params)
        self.gym.prepare_sim(sim)
        return sim

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    #@cached_property
    @property
    def dt(self):
        return self.gym.get_sim_params(self.sim).dt * self.skip_frames

    def get_obs(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def reset_idx(self, env_idx):
        raise NotImplementedError

    def reset(self):
        for e_idx in range(self.num_envs):
            self.reset_idx(e_idx)
        return self.get_obs()

    def render(self, sync_frame_time, env_idx=None, mode='human'):
        #self.gym.fetch_results(self.sim, True)
        if mode == 'rgb_array':
            return self.render_env(sync_frame_time, env_idx=env_idx)

        elif mode == 'human':
            # update the viewer
            self.gym.step_graphics(self.sim)

            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)

            if not self.viewer:
                return

            self.gym.draw_viewer(self.viewer, self.sim, True)

    def render_env(self, sync_frame_time, env_idx=None):
        camera_handle = self.camera_handles.get(env_idx)
        env = self.envs[env_idx]

        if camera_handle is None:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 512
            camera_props.height = 512
            #camera_props.far_plane = 5.
            #camera_props.near_plane = 0.
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(-2.5, -0., 2.5)
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
            actor_name = self.gym.get_actor_name(env, self.actor_handles[env_idx])
            body_handle = self.gym.get_rigid_handle(env, actor_name, 'torso')

            self.gym.attach_camera_to_body(
                camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM
            )

            self.camera_handles[env_idx] = camera_handle

        # update the viewer
        self.gym.step_graphics(self.sim)

        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        im = self.gym.get_camera_image(self.sim, env, camera_handle, gymapi.IMAGE_COLOR)

        return im.reshape(512, 512, -1).astype(np.float32) / 255.


class QuantrupedIsaac(IsaacEnv):
    
    ASSET_ROOT = './simulation_envs/assets'
    ASSET_FILE = 'ant_isaac.xml'

    OBS_FIELDS = [ 
        'body_height',                     # Index 0-4
        'body_qpos_x', 'body_qpos_y',
        'body_qpos_z', 'body_qpos_w',

        'fl_hip', 'fl_knee',               # Index 5-12
        'hl_hip', 'hl_knee',
        'hr_hip', 'hr_knee',
        'fr_hip', 'fr_knee',
                                           # Index 13-18
        'body_vel_x', 'body_vel_y', 'body_vel_z',
        'body_rot_vel_x', 'body_rot_vel_y', 'body_rot_vel_z',

        'fl_hip_vel', 'fl_knee_vel',       # Index 19-26
        'hl_hip_vel', 'hl_knee_vel', 
        'hr_hip_vel', 'hr_knee_vel',
        'fr_hip_vel', 'fr_knee_vel', 

        'fl_hip_pforce', 'fl_knee_pforce', # Index 27-34
        'hl_hip_pforce', 'hl_knee_pforce', 
        'hr_hip_pforce', 'hr_knee_pforce',
        'fr_hip_pforce', 'fr_knee_pforce', 
                                           # Index 35-42
        'fl_hip_hist_ctrl', 'fl_knee_vel_hist_ctrl', 
        'hl_hip_hist_ctrl', 'hl_knee_vel_hist_ctrl', 
        'hr_hip_hist_ctrl', 'hr_knee_vel_hist_ctrl',
        'fr_hip_hist_ctrl', 'fr_knee_vel_hist_ctrl', 
    ]

    ACTION_FIELDS = [
        'fl_hip', 'fl_knee',
        'hl_hip', 'hl_knee',
        'hr_hip', 'hr_knee',
        'fr_hip', 'fr_knee',
    ]

    # contact forces are excerted onto bodies, not joints
    CONTACT_FORCE_FIELDS = [
        #'body_floor', # ToDO: this is slightly different in isaac gym - check if it is valid to just exclude this first field
        'body',
        'fl_hip', 'fl_leg', 'fl_foot',
        'hl_hip', 'hl_leg', 'hl_foot',
        'hr_hip', 'hr_leg', 'hr_foot',
        'fr_hip', 'fr_leg', 'fr_foot'
    ]

    # TODO:
    # X get_action_indices
    # X get_contact_force_indices
    # X get_obs_indices
    # X compute reward
    # X add noise to reset
    # X compute if done
    # * make env for target velocity
    # * make env for bumpy terrain
    # * include com-based forces as a penalty 
    #   (see. cfrc_ext in https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=cfrc_ext#mjdata)

    def __init__(self, ctrl_cost_weight=0.5, contact_cost_weight=5e-4, healthy_reward=0., hf_smoothness=1., **kwargs):
        super().__init__(**kwargs)
        self.envs = []
        self.actor_handles = []
        self._create_floor_plain()

        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.healthy_reward = healthy_reward
        self.hf_smoothness = hf_smoothness
        self.healthy_z_range = (0.2, 1.0)
        self._healthy_reward = 1.0
        self.reset_noise_scale = 0.1
        # last action is part of the observations, this variable holds last actions
        self.last_actions = None
        self.max_steps = 1000
        self.step_counter = 0
        self.frame_skip = 5
        self.target_vel = None

        for _ in range(self._num_envs):
            self._create_env()

    #@lru_cache(maxsize=1)
    def get_body_states(self, time):
        return self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL).reshape(self.num_envs, -1)

    #@lru_cache(maxsize=1)
    def get_dof_states(self, time):
        return self.gym.get_vec_actor_dof_states(self.envs, self.actor_handles, gymapi.STATE_ALL)

    #@lru_cache(maxsize=1)
    def get_dof_forces(self, time):
        forces = [ self.gym.get_actor_dof_forces(e, a)
                   for e, a in zip(self.envs, self.actor_handles) ]
        return np.array(forces)

    #@lru_cache(maxsize=1)
    def _get_contact_forces(self, time):
        return self.gym.get_rigid_contact_forces(self.sim).reshape(self.num_envs, -1)

    def get_contact_forces(self):
        #return self.__to_dict(structured_to_unstructured(self.gym.get_rigid_contact_forces(self.sim).reshape(self.num_envs, -1)))
        return self.__to_dict(self._contact_force_buffer)

    @property
    def done(self):
        return np.logical_not(self.is_healthy(self.time)) #if self._terminate_when_unhealthy else False

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def time(self):
        return self.gym.get_sim_time(self.sim)

    @classmethod
    def observation_space(cls):
        return spaces.Box(-np.inf, np.inf, (len(cls.OBS_FIELDS),), np.float64)

    @property
    def action_space(self):
        highs = []
        lows = []

        for e, a in zip(self.envs, self.actor_handles):
            actuator_properties = self.gym.get_actor_actuator_properties(e, a)
            l = [ ap.lower_control_limit for ap in actuator_properties ]
            h = [ ap.upper_control_limit for ap in actuator_properties ]
            lows.append(l)
            highs.append(h)

        return np.array(lows), np.array(highs)

    def set_target_velocity(self, t_vel):
        if self.target_vel is None:
            # target velocity has never been set
            # append target velocity as last index of the observation
            self.OBS_FIELDS.append('body_target_x_vel')
        self.target_vel = np.array(t_vel)[...,np.newaxis]

    #@lru_cache(maxsize=1)
    def is_healthy(self, time):
        torso = structured_to_unstructured(self.get_body_states(self.time))
        torso_vel = torso[:,0, 7:10] # accesses [:,0]['vel']['linear']
        torso_pos = torso[:,0, :3]   # accesses [:,0]['vel']['linear']
        
        # adapted from [1]
        min_z, max_z = self.healthy_z_range

        return np.logical_and(
            np.isfinite(torso_pos).all() and np.isfinite(torso_vel).all(),
            (min_z <= torso_pos[:,2]), 
            (torso_pos[:,2] <= max_z))

    def _init_actor_transform(self):
        pos = gymapi.Transform()
        pos.p = gymapi.Vec3(0.0, 0.0, 0.44)
        return pos

    def _create_env(self):
        ant_asset = self.gym.load_asset(self.sim, self.ASSET_ROOT, self.ASSET_FILE)
        self.gears = [ dp.motor_effort for dp in self.gym.get_asset_actuator_properties(ant_asset) ]


        env_spacing = 4.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        envs_per_row = int(np.sqrt(self.num_envs+1))

        env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
        actor = self.gym.create_actor(env, ant_asset, self._init_actor_transform(), f'Ant-{len(self.envs)}', 1)
        self.envs.append(env)
        self.actor_handles.append(actor)

        color = gymapi.Vec3(np.random.uniform(), np.random.uniform(), np.random.uniform())
        for rb in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_color(env, actor, rb, gymapi.MESH_VISUAL_AND_COLLISION, color)

        self.body_state_dtype = self.get_body_states(-1).dtype
        self.dof_state_dtype = self.get_dof_states(-1).dtype
        self.n_rigid_bodies = int(self.gym.get_sim_rigid_body_count(self.sim) / self.num_envs)

        # save initial state of rigid bodies and DOFs for resetting
        self.init_body_state = structured_to_unstructured(self.get_body_states(-1)).copy()
        self.init_dof_state = structured_to_unstructured(self.get_dof_states(-1)).copy()
        
        n_actuators = self.gym.get_actor_actuator_count(self.envs[0], self.actor_handles[0])

        if self.last_actions is None:
            self.last_actions = np.zeros([len(self.envs), n_actuators])
        else:
            new_last_actions = len(self.envs) - len(self.last_actions)
            new_last_actions = np.zeros([new_last_actions, n_actuators])
            self.last_actions = np.concatenate((
                self.last_actions,
                new_last_actions),
                axis=0)

    def _create_floor_plain(self):
        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def scale_mass(self, scale):
        
        # iter environments
        for e_handle in self.envs:
            # iter agents of environment
            for actor_idx in range(self.gym.get_actor_count(e_handle)):
                actor_handle = self.gym.get_actor_handle(e_handle, actor_idx)
                body_props = self.gym.get_actor_rigid_body_properties(e_handle, actor_handle)
                # scale the mass of all rigit bodies
                for bp in body_props:
                    bp.mass *= scale
                # apply new rigit body properties
                self.gym.set_actor_rigid_body_properties(e_handle, actor_handle, body_props, True)

    def scale_mass_env(self, env_id, scale):
        
        # iter agents of environment
        for actor_idx in range(self.gym.get_actor_count(self.envs[env_id])):
            actor_handle = self.gym.get_actor_handle(self.envs[env_id], actor_idx)
            body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], actor_handle)
            # scale the mass of all rigit bodies
            for bp in body_props:
                bp.mass *= scale
            # apply new rigit body properties
            self.gym.set_actor_rigid_body_properties(self.envs[env_id], actor_handle, body_props, True)

    def iter_by_env(self, arr):
        for i in range(self.num_envs):
            for e in arr:
                yield e + f'_{i}'

    def step(self, actions):
        dof_efforts = actions * self.gears
        pre_sim_body_states = self.get_body_states(self.time).copy()
        #pre_sim_dof_states = self.get_dof_states(self.time)

        # apply the forces
        for i in range(len(dof_efforts)): #range(len(self.envs)):
            self.gym.apply_actor_dof_efforts(self.envs[i], self.actor_handles[i], list(dof_efforts[i]))
        
        # step the physics
        for s in range(self.frame_skip):
            self.gym.simulate(self.sim)
            if s == 0:
                self._contact_force_buffer = self._get_contact_forces(self.time)
                continue
            # structured arrays cannot be added, concatenation is a fast option here
            # calling structured_to_unstructured in this loop takes very long
            self._contact_force_buffer = np.concatenate((
                self._contact_force_buffer, 
                self._get_contact_forces(self.time)))
        
        self._contact_force_buffer = structured_to_unstructured(self._contact_force_buffer)\
            .reshape(self.skip_frames, self.num_envs, self.n_rigid_bodies, 3).sum((0,-1))

        self.gym.fetch_results(self.sim, True)

        post_sim_body_states = self.get_body_states(self.time).copy()
        # update last actions
        self.last_actions = actions
        # then compute observation
        obs = self._get_obs()
        rewards, info = self._get_reward(actions, pre_sim_body_states, post_sim_body_states)
        #contact_forces = structured_to_unstructured(self._get_contact_forces(self.time))

        info['contact_forces'] = self.__to_dict(self._contact_force_buffer)

        '''
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
        '''
        self.step_counter += 1
        done_list = np.logical_or(self.done, self.step_counter >= self.max_steps)
        dones = self.__to_dict(done_list)
        dones["__all__"] = all(done_list)
        
        return obs, rewards, dones, info

    def __to_dict(self, arr):
        return dict(enumerate(arr))

    def _get_agent_reward(self, agent_idx, action, prev_state, post_state):
        # calculate forward velocity reward
        torso_prev_pos, _ = prev_state[0]
        torso_post_pos, _  = post_state[0]
        torso_x_prev = torso_prev_pos[0][0]
        torso_x_post = torso_post_pos[0][0]
        torso_x_vel = (torso_x_post - torso_x_prev)/self.gym.get_sim_params(self.sim).dt
        # calculate reward for being a healthy robot
        healthy_reward = float(self.is_healthy(self.time)[agent_idx]) * self._healthy_reward
        # calculate energy cost of the current action
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        return torso_x_vel + healthy_reward - control_cost

    def get_forward_reward(self, prev_states, post_states):
        prev_torso_x_pos = prev_states[:,0]['pose']['p']['x']
        post_torso_x_pos = post_states[:,0]['pose']['p']['x']
        torso_delta_x = (post_torso_x_pos - prev_torso_x_pos)
        torso_x_vel = torso_delta_x / self.dt

        if self.target_vel is not None:
            vel_diff = torso_x_vel - self.target_vel[:,0]
            # for faster walking scale down the reward
            vel_scale = 1 + 1. / self.target_vel[:,0]
            # velocity reward
            vel_reward = 1. / (np.abs(vel_diff) + 1)
            # this is the distance from maximum reward to 1., or alternatively max(reward) - 1
            min_reward = vel_scale * 1. / (self.target_vel[:,0] + 1)
            return vel_scale * vel_reward - min_reward

        return torso_x_vel

    def _get_reward(self, actions, prev_states, post_states):
        # compute the reward for moving forward
        forward_reward = self.get_forward_reward(prev_states, post_states)
        # calculate reward for being a healthy robot
        healthy_reward = self.is_healthy(self.time) * self._healthy_reward
        # calculate energy cost of the current action
        total_control_cost = self.ctrl_cost_weight * np.sum(np.square(actions), axis=-1)
        contacts = structured_to_unstructured(self._get_contact_forces(self.time))
        contacts = np.square(np.clip(contacts, -1., 1.))
        contact_cost = self.contact_cost_weight * contacts
        total_contact_cost = np.sum(contact_cost, axis=(-1, -2))

        reward_info = {
                'forward_reward' : self.__to_dict(forward_reward),
                'healthy_reward' : self.__to_dict(healthy_reward),
                'total_control_cost' : self.__to_dict(total_control_cost),
                'total_contact_cost' : self.__to_dict(total_contact_cost),
                'control_costs' : self.__to_dict(np.square(actions)),
                'contact_cost' : self.__to_dict(contact_cost),
        }

        reward = forward_reward + healthy_reward - total_contact_cost - total_control_cost

        return self.__to_dict(reward), reward_info

    def _get_agent_obs(self, agent_idx):
        rigid_body_states = self.get_body_states(self.time)[agent_idx]
        dof_states = self.get_dof_states(self.time)[agent_idx]
        dof_forces = self.get_dof_forces(self.time)[agent_idx]
        
        torso = rigid_body_states[0]
        torso_height = torso[0][0][2]
        torso_quat   = torso[0][1]
        torso_vel    = torso[1][0]
        torso_angular_vel = torso[1][1]
        
        fl_hip_pos,   fl_hip_vel   = dof_states[0]
        fl_ankle_pos, fl_ankle_vel = dof_states[1]
        hl_hip_pos,   hl_hip_vel   = dof_states[2]
        hl_ankle_pos, hl_ankle_vel = dof_states[3]
        hr_hip_pos,   hr_hip_vel   = dof_states[4]
        hr_ankle_pos, hr_ankle_vel = dof_states[5]
        fr_hip_pos,   fr_hip_vel   = dof_states[6]
        fr_ankle_pos, fr_ankle_vel = dof_states[7]

        if self.last_actions is None:
            last_actions = np.zeros_like(dof_forces)
        else:
            last_actions = self.last_actions[agent_idx]

        return np.array([
            torso_height, *torso_quat, 
            fl_hip_pos, fl_ankle_pos, hl_hip_pos, hl_ankle_pos,
            hr_hip_pos, hr_ankle_pos, fr_hip_pos, fr_ankle_pos,
            *torso_vel, *torso_angular_vel,
            fl_hip_vel, fl_ankle_vel, hl_hip_vel, hl_ankle_vel,
            hr_hip_vel, hr_ankle_vel, fr_hip_vel, fr_ankle_vel,
            *dof_forces,
            *last_actions
        ])
        
    def _get_obs(self):
        # converting the full-size array is fastest than converting multiple smaller
        body_states = structured_to_unstructured(self.get_body_states(self.time))
        dof_states = self.get_dof_states(self.time)
        dof_forces = self.get_dof_forces(self.time)

        #torso_heights  = body_states['pose']['p']['z'][:,0]
        #torso_quats    = body_states['pose']['r'][:,0]
        #torso_lin_vels = body_states['vel']['linear'][:,0]
        #torso_ang_vels = body_states['vel']['angular'][:,0]
        torso_heights  = body_states[:,0,2]
        torso_quats    = body_states[:,0,3:7]
        torso_lin_vels = body_states[:,0,7:10]
        torso_ang_vels = body_states[:,0,10:]
        body_states = self.get_body_states(self.time)
        dof_states = self.get_dof_states(self.time)
        dof_forces = self.get_dof_forces(self.time)

        leg_dof_positions  = dof_states['pos']
        leg_dof_velocities = dof_states['vel']

        if self.last_actions is None:
            last_actions = np.zeros_like(dof_forces)
        else:
            last_actions = self.last_actions

        obs = np.concatenate((
            torso_heights[...,np.newaxis],
            torso_quats,
            leg_dof_positions,
            torso_lin_vels,
            torso_ang_vels,
            leg_dof_velocities,
            dof_forces, last_actions,
        ), axis=-1).astype(np.float32)

        if self.target_vel is not None:
            obs = np.concatenate((obs, self.target_vel), axis=-1)
            
        return self.__to_dict(obs)

    def reset_env(self, env_id, return_obs=False):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        body_pos_noise = np.random.uniform(
            low=noise_low, 
            high=noise_high, 
            size=self.init_body_state.shape[1:])
        body_vel_noise = np.random.normal(
            size=self.init_body_state.shape[1:]) * self.reset_noise_scale

        init_bodies_pos = self.init_body_state[env_id] + body_pos_noise
        init_bodies_vel = self.init_body_state[env_id] + body_pos_noise
        init_bodies = unstructured_to_structured(init_bodies_pos, 
            dtype=self.body_state_dtype)
        init_bodies['vel'] = unstructured_to_structured(init_bodies_vel, 
            dtype=self.body_state_dtype)['vel']
        #nit_bodies = unstructured_to_structured(self.init_body_state, 
        #    dtype=self.body_state_dtype)

        dof_pos_noise = np.random.uniform(
            low=noise_low, 
            high=noise_high, 
            size=self.init_dof_state.shape[1:])
        dof_vel_noise = np.random.normal(
            size=self.init_dof_state.shape[1:]) * self.reset_noise_scale

        init_dofs_pos = self.init_dof_state[env_id] + dof_pos_noise
        init_dofs_vel = self.init_dof_state[env_id] + dof_pos_noise
        init_dofs = unstructured_to_structured(init_dofs_pos, 
            dtype=self.dof_state_dtype)
        init_dofs['vel'] = unstructured_to_structured(init_dofs_vel, 
            dtype=self.dof_state_dtype)['vel']

        self.gym.set_actor_rigid_body_states(
            self.envs[env_id], 
            self.actor_handles[env_id], 
            init_bodies, 
            gymapi.STATE_ALL)

        self.gym.set_actor_dof_states(
            self.envs[env_id], 
            self.actor_handles[env_id], 
            init_dofs[env_id], 
            gymapi.STATE_ALL)
        self.last_actions[env_id] = 0.
        
        #self.clear_cache()
        if return_obs:
            return self._get_agent_obs(env_id)

    def reset(self):
        self.step_counter = 0
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        body_pos_noise = np.random.uniform(
            low=noise_low, 
            high=noise_high, 
            size=self.init_body_state.shape)
        body_vel_noise = np.random.normal(
            size=self.init_body_state.shape) * self.reset_noise_scale

        init_bodies_pos = self.init_body_state + body_pos_noise
        init_bodies_vel = self.init_body_state + body_pos_noise
        init_bodies = unstructured_to_structured(init_bodies_pos, 
            dtype=self.body_state_dtype)
        init_bodies['vel'] = unstructured_to_structured(init_bodies_vel, 
            dtype=self.body_state_dtype)['vel']
        #nit_bodies = unstructured_to_structured(self.init_body_state, 
        #    dtype=self.body_state_dtype)

        dof_pos_noise = np.random.uniform(
            low=noise_low, 
            high=noise_high, 
            size=self.init_dof_state.shape)
        dof_vel_noise = np.random.normal(
            size=self.init_dof_state.shape) * self.reset_noise_scale

        init_dofs_pos = self.init_dof_state + dof_pos_noise
        init_dofs_vel = self.init_dof_state + dof_pos_noise
        init_dofs = unstructured_to_structured(init_dofs_pos, 
            dtype=self.dof_state_dtype)
        init_dofs['vel'] = unstructured_to_structured(init_dofs_vel, 
            dtype=self.dof_state_dtype)['vel']

        self.gym.set_sim_rigid_body_states(self.sim, init_bodies, gymapi.STATE_ALL)

        for i, (e, a) in enumerate(zip(self.envs, self.actor_handles)):
            self.gym.set_actor_dof_states(e, a, init_dofs[i], gymapi.STATE_ALL)

        self.last_actions = np.zeros_like(self.get_dof_forces(self.time))
        obs = self._get_obs()

        #self.clear_cache()

        return obs

    def clear_cache(self):
        self.get_body_states.cache_clear()
        #self.get_dof_states.cache_clear()
        #self.get_dof_forces.cache_clear()

    @classmethod
    def get_obs_indices(cls, prefixes=None):
        '''
        Returns the indices for the observations starting with one of the
        given prefixes.
        '''
        obs_indices = []

        # if no prefixes are given, pass an array with all indices.
        if prefixes is None:
            return np.arange(len(cls.OBS_FIELDS))

        # this respects the ordering as is in prefixes, e.g.
        # if prefixes = ['body', 'hl'], the first indices of an observation
        # are populated with body features and the last with left-hindleg features. 
        for prefix in prefixes:
            idx = [ f.startswith(prefix) for f in cls.OBS_FIELDS ]
            obs_indices.extend(list(np.where(idx)[0]))

        return obs_indices

    @classmethod
    def get_action_indices(cls, prefixes=None):
        '''
        Returns the indices for the actions starting with one of the
        given prefixes.
        '''
        action_indices = []

        # if no prefixes are given, pass an array with all indices.
        if prefixes is None:
            return np.arange(len(cls.ACTION_FIELDS))

        for prefix in prefixes:
            idx = [ f.startswith(prefix) for f in cls.ACTION_FIELDS ]
            action_indices.extend(list(np.where(idx)[0]))

        return action_indices

    @classmethod
    def get_contact_force_indices(cls, prefixes=None, weights=None):
        '''
        Returns the indices for the contact_forces starting with one of the
        given prefixes.
        '''
        contact_force_indices = []
        contact_force_weights = []

        # if no prefixes are given, pass an array with all indices.
        if prefixes is None:
            n_fields = len(cls.CONTACT_FORCE_FIELDS)
            return np.arange(n_fields), np.ones([n_fields, 1])

        if weights is None:
            weights = np.ones(len(prefixes))

        for prefix, weight in zip(prefixes, weights):
            mask = [ f.startswith(prefix) for f in cls.CONTACT_FORCE_FIELDS ]
            idx = list(np.where(mask)[0])
            contact_force_indices.extend(idx)
            contact_force_weights.extend([[weight]]*len(idx))

        return contact_force_indices, contact_force_weights
