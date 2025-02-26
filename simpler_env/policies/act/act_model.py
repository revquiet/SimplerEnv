import cv2
import logging
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision import transforms as v2

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from own.test import WorldModel
from own.bridge import Bridge
from own.policy import ACTBridge
from own.data.data_utils import *

logging.basicConfig(level=logging.DEBUG)

class ILEvaluator:
    def __init__(self, config: DictConfig, policy_path: Path):
        self.bridge = Bridge(config)
        self.policy = ACTBridge(model_path=policy_path)
        self.world = WorldModel(load_upsample=True)

        self.bridge.reset_robot()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_robot_action(action, origin_left_ee, origin_right_ee, new_left_ee, new_right_ee):
        left_hand_position = new_left_ee[:3]-origin_left_ee[:3]
        left_hand_rotation = new_left_ee[3:]-origin_left_ee[3:]
        right_hand_position = new_right_ee[:3]-origin_right_ee[:3]
        right_hand_rotation = new_right_ee[3:]-origin_right_ee[3:]
        action = Pose(
            left_hand_position=left_hand_position,
            left_hand_rotation=left_hand_rotation,
            right_hand_position=right_hand_position,
            right_hand_rotation=right_hand_rotation,
        )
        norm_action = normalize_speed_pose(action)
        return norm_action

    def norm_action(self, action):
        origin_left_ee, origin_right_ee = self.bridge.get_ee_pose()
        origin_left_hand, origin_right_hand = self.bridge.robot.get_hand_real_pose()
        self.bridge.step(action)
        new_left_ee, new_right_ee = self.bridge.get_ee_pose()
        norm_robot_action = self.encode_robot_action(origin_left_ee, origin_right_ee, new_left_ee, new_right_ee)
        norm_action = add_hand_norm_speed_data(norm_robot_action, np.concatenate([origin_left_hand, origin_right_hand]), action[:12])
        return norm_action
    
    def maintain_aspect_ratio_resize(self, image, target_size):
        target_h, target_w = target_size[1], target_size[0]
        height, width = image.shape[:2]
        
        # Calculate aspect ratio of original and target size
        aspect_ratio_orig = width / height
        aspect_ratio_target = target_w / target_h
        
        if aspect_ratio_orig > aspect_ratio_target:
            # Image is too wide, crop in the width direction
            new_width = int(height * aspect_ratio_target)
            start_x = (width - new_width) // 2
            image = image[:, start_x:start_x+new_width]
        elif aspect_ratio_orig < aspect_ratio_target:
            # Image is too tall, crop in the height direction
            new_height = int(width / aspect_ratio_target)
            start_y = (height - new_height) // 2
            image = image[start_y:start_y+new_height]
        
        # Resize the image to the target size
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.transpose(2, 0, 1)
        return resized_image
    
    def transform_image(self, img, crop_size=(200,  200)):
        img = img / 255.0
        img_reshape = self.maintain_aspect_ratio_resize(img, crop_size)
        logging.debug(f'reshape img shape:{img_reshape.shape}')
        img_reshape = torch.from_numpy(img_reshape).to(torch.float32)

        patch_h = 16
        patch_w = 22
        transform = v2.Compose(
                    [
                        v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    ]
                )
        
        transform_img = transform(img_reshape)
        return transform_img

    def step(self, observation):
        action = self.policy.Predict(observation)
        norm_action = self.norm_action(action)
        new_bos = self.world.step(norm_action)
        
        new_bos = new_bos.squeeze().cpu().numpy()
        new_bos = new_bos.transpose(1, 2, 0)
        print(f'new_bos shape:{new_bos.shape}')
        return new_bos
    
    def run(self):
        step = 0
        #(720, 1280, 3)
        image = obs["image"]["3rd_view_camera"]["rgb"]
        image = image.transpose(1, 2, 0)
        # obs = self.world.get_init_observation()
        # obs = obs.transpose(1, 2, 0)
        print(f'get init obs shape:{obs.shape}')
        states = None



        
        while True:
            # cv2.imwrite(f'/home/ubuntu/code/test/diamond/src/own/outputs/debug/{step}.jpg', obs)
            states = self.bridge.observe()
            print(f'states shape:{states.shape}')
            input_img = self.transform_image(obs, crop_size=(200, 200))
            print(f'input_img shape:{input_img.shape}')
            states = torch.from_numpy(states).to(torch.float32)

            states = states.to(self.device, non_blocking=True)
            image = input_img.to(self.device, non_blocking=True)
            states = states.unsqueeze(0)
            image = image.unsqueeze(0)

            observation = {
                "observation.state": states,
                "observation.image.left": image,
            }

            obs = self.step(observation)
            print(f'obs shape:{obs.shape}')
            step += 1

def main(config):
    # act_path = '/home/ubuntu/code/test/fourier-lerobot/outputs/train/2025-01-13/13-55-33_fourier_act_act_pick_and_place/checkpoints/200000/pretrained_model'
    act_path = '/home/fftai/Code/git_repo/fourier-lerobot/outputs/train/2025-01-13/18-51-39_real_world_act_pick_and_place/checkpoints/190000/pretrained_model'
    il_eval = ILEvaluator(policy_path=Path(act_path))
    il_eval.run()
    return

if __name__=='__main__':
    main()
