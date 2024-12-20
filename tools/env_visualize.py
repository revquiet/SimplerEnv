import mani_skill2_real2sim.envs, gymnasium as gym
import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose


# env = gym.make(
#     "PutEggplantInBasketScene-v0",
#     # "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     # num_envs=1,
#     obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
#     control_mode="arm_pd_ee_target_delta_pose_gripper_pd_joint_pos", # there is also "pd_joint_delta_pos", ...
#     render_mode="human"
# )
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(terminated, truncated)
#     # done = terminated or truncated
#     env.render()  # a display is required to render
# env.close()

env3 = gym.make('CloseDrawerCustomInScene-v0', obs_mode='rgbd', 
    robot='google_robot_static', sim_freq=513, control_freq=3, max_episode_steps=113, 
    control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
    scene_name='bridge_table_1_v1', 
    station_name='mk_station_recolor', # cabinet model
    shader_dir='rt', # enable raytracing, slow for non RTX gpus
    render_mode="human"
)
obs3, reset_info_3 = env3.reset(options={
    'robot_init_options': {
        'init_xy': np.array([0.75, 0.00]),
    },
})
instruction3 = env3.get_language_instruction()
image3 = obs3['image']['overhead_camera']['rgb']
done = False
while not done:
    action = env3.action_space.sample()
    obs, reward, terminated, truncated, info = env3.step(action)
    print(terminated, truncated)
    # done = terminated or truncated
    env3.render()  # a display is required to render
env3.close()
