import argparse
import os
import numpy as np
import torch
from pathlib import Path
import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict, transform_image
from lerobot.common.policies.act.modeling_act import ACTPolicy
from simpler_env.utils.hand_retarget.hand import HandRetarget
import mediapy as media
from omegaconf import OmegaConf, DictConfig
import time
# 获取当前文件所在目录的绝对路径

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设 tools 目录就在根目录下）
project_root = os.path.abspath(os.path.join(current_dir, "../"))
# 添加到 sys.path
import sys
if project_root not in sys.path:
    sys.path.append(project_root)
from tools.sysid.rot_trans import convert_quaternion_to_rotation_matrix, convert_rotation_matrix_to_euler
from tools.data_visualize import rerun_vis

# Function to load policy
def load_policy(policy_path, policy_mode='act'):
    pretrained_policy_path = Path(policy_path)
    if policy_mode == 'act':
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    else:
        raise ValueError(f'Invalid policy mode input: {policy_mode}, only act mode support')
    policy.eval()
    return policy

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="act", choices=["act", "rt1", "octo-base", "octo-small"])
parser.add_argument("--ckpt-path", type=str, default="/home/fftai/Code/git_repo/fourier-lerobot/outputs/train/2025-01-13/18-51-39_real_world_act_pick_and_place/checkpoints/190000/pretrained_model")
# parser.add_argument("--ckpt-path", type=str, default="/home/fftai/Downloads/13-55-33_fourier_act_act_pick_and_place/checkpoints/170000/pretrained_model")
parser.add_argument("--task", default="grx_robot_carrot_on_plate", choices=ENVIRONMENTS)
parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
parser.add_argument("--n-trajs", type=int, default=10)
args = parser.parse_args()

# Initialize policy
policy = load_policy(args.ckpt_path, policy_mode='act')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)

# Initialize hand retargeting
cfg = OmegaConf.load("simpler_env/utils/hand_retarget/configs/hand/fourier.yaml")
hand_retarget = HandRetarget(cfg)
left_landmarks = np.zeros([21,3])
right_landmarks = np.zeros([21,3])
left_qpos, right_qpos = hand_retarget.retarget(left_landmarks, right_landmarks)

# Initialize environment
env = simpler_env.make(args.task)

# Inference loop
success_arr = []
for ep_id in range(args.n_trajs):
    # if ep_id != 2:
    #     continue
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()

    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    policy.reset()

    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    input_image = transform_image(img=image, crop_size=(400, 240))
    images = [image]
    input_images = [np.array(input_image).transpose(1, 2, 0)]  
    predicted_terminated, success, truncated = False, False, False
    timestep = 0
    list_left_arm_action = []
    list_right_arm_action = []

    while not (predicted_terminated or truncated):
        left_arm_qpos = np.zeros(7)
        right_arm_qpos = obs['agent']['qpos'][0:7]

        left_hand_qpos = np.zeros(6)
        _, right_hand_qpos_11dof = hand_retarget.sapien_to_hand_11dof(np.zeros(11), obs['agent']['qpos'][7:18])
        _, right_hand_qpos = hand_retarget.qpos_to_real(np.zeros(11), right_hand_qpos_11dof)

        states = np.concatenate([left_hand_qpos, right_hand_qpos, left_arm_qpos, right_arm_qpos])
        print("policy input:\n",right_hand_qpos,"\n",right_arm_qpos)
        action = policy.select_action({
            "observation.state": torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device),
            "observation.image.left": torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).to(device),
        })

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()
        left_hand_action = numpy_action[0:6]
        right_hand_action = numpy_action[6:12]
        left_arm_action = numpy_action[12:19]
        right_arm_action = numpy_action[19:26]

        print("policy output:\n",right_hand_action,"\n",right_arm_action)

        # ----------------------------------------
        # arm
        prev_ee_pose_at_base = env.agent.controller.controllers['arm'].compute_fk(right_arm_qpos)
        current_ee_pose_at_base = env.agent.controller.controllers['arm'].compute_fk(right_arm_action)

        action_world_vector = current_ee_pose_at_base.p - prev_ee_pose_at_base.p
        rotation_matrix_before = convert_quaternion_to_rotation_matrix((prev_ee_pose_at_base.q)[[1, 2, 3, 0]]).T
        rotation_matrix_next = convert_quaternion_to_rotation_matrix((current_ee_pose_at_base.q)[[1, 2, 3, 0]])
        rotation_diff = np.dot(rotation_matrix_next , rotation_matrix_before)
        action_rotation_delta = convert_rotation_matrix_to_euler(rotation_diff)

        # hand
        _, right_hand_action_qpos = hand_retarget.real_to_qpos(np.zeros(6), right_hand_action)
        _, right_hand_action_11dof =hand_retarget.from_6dof_to_11dof(np.zeros(6), right_hand_action_qpos)
        _, right_hand_action_sapien = hand_retarget.hand_to_sapien_11dof(np.zeros(11), right_hand_action_11dof)

        action = {}
        action["action_world_vector"] = action_world_vector
        action["action_rotation_delta"] = action_rotation_delta
        action["action_gripper"] = right_hand_action_sapien
        # print("sapien_input:\n",action["action_world_vector"], action["action_rotation_delta"])
        # ----------------------------------------
        # Perform environment step with the action
        obs, reward, success, truncated, info = env.step(np.concatenate([action["action_world_vector"], action["action_rotation_delta"], action["action_gripper"]]))

        print(timestep,"-------------------")
        # Update the instruction for long horizon tasks
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction
            print(instruction)
        is_final_subtask = env.is_final_subtask()

        # Update image observation
        image = get_image_from_maniskill2_obs_dict(env, obs)
        input_image = transform_image(img=image, crop_size=(400, 240))
        images.append(image)
        input_images.append(np.array(input_image).transpose(1, 2, 0))
        list_left_arm_action.append(np.concatenate([prev_ee_pose_at_base.p, prev_ee_pose_at_base.q]))
        list_right_arm_action.append(np.concatenate([current_ee_pose_at_base.p, current_ee_pose_at_base.q]))
        timestep += 1
        # time.sleep(0.2)

    # Save episode video
    episode_stats = info.get("episode_stats", {})
    success_arr.append(success)
    print(f"Episode {ep_id} success: {success}")
    media.write_video(f"{args.logging_root}/episode_{ep_id}_success_{success}.mp4", images, fps=60)
    media.write_video(f"{args.logging_root}/input/episode_{ep_id}_success_{success}.mp4", input_images, fps=60)
    rerun_vis(list_left_arm_action,list_right_arm_action)

# Output overall success rate
print(
    "**Overall Success**",
    np.mean(success_arr),
    f"({np.sum(success_arr)}/{len(success_arr)})",
)
