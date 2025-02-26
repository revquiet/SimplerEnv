import cv2
import logging
import torch
import torchvision.transforms as v2

def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        elif "grx_robot" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]

def maintain_aspect_ratio_resize(image, target_size):
    """
    Resize image while maintaining the aspect ratio, crop if necessary
    """
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

def transform_image(img, crop_size=(200,  200)):
    # Preprocess the image for passing it to the pretrained Dino_v2 model
    img = img / 255.0
    img_reshape = maintain_aspect_ratio_resize(img, crop_size)
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