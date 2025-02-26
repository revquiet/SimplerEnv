import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from omegaconf import DictConfig, OmegaConf

from simpler_env.utils.hand_retarget.retarget_utils import ASSET_DIR, remap


class HandRetarget:
    def __init__(self, cfg: DictConfig) -> None:
        assets_dir = ASSET_DIR
        RetargetingConfig.set_default_urdf_dir(assets_dir)
        left_retargeting_config = RetargetingConfig.from_dict(cfg=OmegaConf.to_container(cfg["left"], resolve=True))
        right_retargeting_config = RetargetingConfig.from_dict(cfg=OmegaConf.to_container(cfg["right"], resolve=True))
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()
        self.hand_type = cfg.type
        self.tip_indices = cfg.tip_indices
        self.cfg=cfg

    @property
    def left_joint_names(self):
        return self.left_retargeting.joint_names

    @property
    def right_joint_names(self):
        return self.right_retargeting.joint_names
    
    def retarget(self, left_landmarks: np.ndarray, right_landmarks: np.ndarray):
        left_input = left_landmarks[self.tip_indices]
        right_input = right_landmarks[self.tip_indices]

        if self.left_retargeting.optimizer.retargeting_type.lower() == "dexpilot":
            # for dexpilot, we need to calculate vector between each tip
            left_input_processed = []
            for i in range(len(self.tip_indices)):
                for j in range(i + 1, len(self.tip_indices)):
                    left_input_processed.append(left_input[j] - left_input[i])
            for i in range(len(self.tip_indices)):
                left_input_processed.append(left_input[i] - left_landmarks[0])
            left_input = np.array(left_input_processed)

        if self.right_retargeting.optimizer.retargeting_type.lower() == "dexpilot":
            # for dexpilot, we need to calculate vector between each tip
            right_input_processed = []
            for i in range(len(self.tip_indices)):
                for j in range(i + 1, len(self.tip_indices)):
                    right_input_processed.append(right_input[j] - right_input[i])

            for i in range(len(self.tip_indices)):
                right_input_processed.append(right_input[i] - right_landmarks[0])
            right_input = np.array(right_input_processed)

        left_qpos = self.left_retargeting.retarget(left_input)
        right_qpos = self.right_retargeting.retarget(right_input)
        print(self.right_retargeting.joint_names)
        return left_qpos, right_qpos

    # def retarget(self, left_landmarks: np.ndarray, right_landmarks: np.ndarray):
    #     left_qpos = self.left_retargeting.retarget(left_landmarks[self.tip_indices])
    #     right_qpos = self.right_retargeting.retarget(right_landmarks[self.tip_indices])
    #     return left_qpos, right_qpos

    def qpos_to_real(self, left_qpos, right_qpos):
        """Convert hand joint angles to real values passed to the hand SDK"""

        left_qpos_real = remap(
            left_qpos[self.cfg.actuated_indices],
            self.left_retargeting.joint_limits[:, 0],
            self.left_retargeting.joint_limits[:, 1],
            self.cfg.range_max,
            self.cfg.range_min,
        )

        right_qpos_real = remap(
            right_qpos[self.cfg.actuated_indices],
            self.right_retargeting.joint_limits[:, 0],
            self.right_retargeting.joint_limits[:, 1],
            self.cfg.range_max,
            self.cfg.range_min,
        )

        return left_qpos_real, right_qpos_real

    def real_to_qpos(self, left_qpos_real, right_qpos_real):
        """Convert real values passed to the hand SDK to hand joint angles"""
        left_qpos = remap(
            left_qpos_real,
            self.cfg.range_max,
            self.cfg.range_min,
            self.left_retargeting.joint_limits[:, 0],
            self.left_retargeting.joint_limits[:, 1],
        )

        right_qpos = remap(
            right_qpos_real,
            self.cfg.range_max,
            self.cfg.range_min,
            self.right_retargeting.joint_limits[:, 0],
            self.right_retargeting.joint_limits[:, 1],
        )

        return left_qpos, right_qpos
    
    def from_6dof_to_11dof(self, left_qpos_6dof, right_qpos_6dof):
        """Convert 6-DOF hand joint angles to 11-DOF hand joint angles
        order from hdf5 file:
            hand_angle = [a, b, c, d, e, f]
                a: R_index_proximal_joint
                b: R_middle_proximal_joint
                c: R_ring_proximal_joint
                d: R_pinky_proximal_joint
                e: R_thumb_proximal_pitch_joint
                f: R_thumb_proximal_yaw_joint
        order from hand sdk:
        wrong:
            "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
            "R_ring_proximal_joint", "R_ring_intermediate_joint", 
            "R_middle_proximal_joint", "R_middle_intermediate_joint", 
            "R_index_proximal_joint", "R_index_intermediate_joint", 
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint", "R_thumb_distal_joint",
        right:
            'R_index_proximal_joint', 'R_index_intermediate_joint', 
            'R_middle_proximal_joint', 'R_middle_intermediate_joint', 
            'R_pinky_proximal_joint', 'R_pinky_intermediate_joint', 
            'R_ring_proximal_joint', 'R_ring_intermediate_joint', 
            'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_distal_joint'
        """
        mimic_coefficients = [0.974, 1.128, 1.131, 1.143, 1.129]
        mimic_indices = [1, 3 ,5, 7, 10]
        drived_indices = [0, 2, 4, 6, 9]
        reorder_6dof = [0, 1, 3, 2, 4]

        left_qpos_11dof = np.zeros(11)
        left_qpos_11dof[8] = left_qpos_6dof[5]
        left_qpos_11dof[drived_indices] = left_qpos_6dof[reorder_6dof]
        left_qpos_11dof[mimic_indices] = left_qpos_11dof[drived_indices] * mimic_coefficients

        right_qpos_11dof = np.zeros(11)
        right_qpos_11dof[8] = right_qpos_6dof[5]
        right_qpos_11dof[drived_indices] = right_qpos_6dof[reorder_6dof]
        right_qpos_11dof[mimic_indices] = right_qpos_11dof[drived_indices] * mimic_coefficients
        
        return left_qpos_11dof, right_qpos_11dof
    
    def hand_to_sapien_11dof(self, left_qpos, right_qpos):
        """Convert joint angles from hand sdk to sapien
        order from hand sdk:
            wrong:
            "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
            "R_ring_proximal_joint", "R_ring_intermediate_joint", 
            "R_middle_proximal_joint", "R_middle_intermediate_joint", 
            "R_index_proximal_joint", "R_index_intermediate_joint", 
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint", "R_thumb_distal_joint",
            current:
            'R_index_proximal_joint', 'R_index_intermediate_joint', 
            'R_middle_proximal_joint', 'R_middle_intermediate_joint', 
            'R_pinky_proximal_joint', 'R_pinky_intermediate_joint', 
            'R_ring_proximal_joint', 'R_ring_intermediate_joint', 
            'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_distal_joint'
        order from sapien:
            "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_distal_joint",
            "R_index_proximal_joint", "R_index_intermediate_joint", 
            "R_middle_proximal_joint", "R_middle_intermediate_joint", 
            "R_ring_proximal_joint", "R_ring_intermediate_joint", 
            "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
        """
        left_qpos_sdk, right_qpos_sdk = left_qpos[[8, 9, 10, 0, 1, 2, 3, 6, 7, 4, 5]], right_qpos[[8, 9, 10, 0, 1, 2, 3, 6, 7, 4, 5]]
        return left_qpos_sdk, right_qpos_sdk
    
    def sapien_to_hand_11dof(self, left_qpos_sdk, right_qpos_sdk):
        """Convert joint angles from sapien to hand sdk"""
        left_qpos, right_qpos = left_qpos_sdk[[3, 4, 5, 6, 9, 10, 7, 8, 0, 1, 2]], right_qpos_sdk[[3, 4, 5, 6, 9, 10, 7, 8, 0, 1, 2]]
        return left_qpos, right_qpos
    
if __name__=="__main__":
    cfg = OmegaConf.load("/home/fftai/Code/python/teleohand/configs/hand/fourier.yaml")
    hand_retarget = HandRetarget(cfg)
    left_landmarks = np.zeros([21,3])
    right_landmarks = np.zeros([21,3])
    left_qpos, right_qpos = hand_retarget.retarget(left_landmarks, right_landmarks)
    print(f"Left hand joints: {left_qpos}")
    print(f"Right hand joints: {right_qpos}")
    left_qpos_real, right_qpos_real = hand_retarget.qpos_to_real(left_qpos, right_qpos)
    print(f"Left hand joints: {left_qpos_real}")
    print(f"Right hand joints: {right_qpos_real}")

    left_qpos_real = [0, 0, 0, 0, 10.3, 10.3]
    right_qpos_real= [0, 0, 0, 0, 10.3, 10.3]
    left_qpos, right_qpos = hand_retarget.real_to_qpos(left_qpos_real, right_qpos_real)
    print(f"Left hand joints6dof: {left_qpos}")
    print(f"Right hand joints6dof: {right_qpos}")
    left_qpos_11dof,right_qpos_11dof = hand_retarget.from_6dof_to_11dof(left_qpos, right_qpos)
    print(f"Left hand joints11dof: {left_qpos_11dof}")
    print(f"Right hand joints11dof: {right_qpos_11dof}")
    print(hand_retarget.left_joint_names)
    print(hand_retarget.right_joint_names)
    print("-----------------")
    print(hand_retarget.left_retargeting.joint_limits)
    print(hand_retarget.right_retargeting.joint_limits)
    r_11dof = np.array([-0.14794728, -0.14419447, -0.14642914, -0.16529682, -0.1480777, -0.1672851, -0.11794896, -0.13475333, 1.031945, -0.8426851, 1.1069062])
    l_11dof = np.zeros(11)
    left_qpos_real, right_qpos_real = hand_retarget.qpos_to_real(l_11dof, r_11dof)
    print(f"Left hand joints-----------: {left_qpos_real}")
    print(f"Right hand joints-----------: {right_qpos_real}")