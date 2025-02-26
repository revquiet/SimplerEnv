import time

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import math

def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 500.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True

    robot: sapien.Articulation = loader.load(
        # "ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/grx_description/GR1T2/urdf/GR1T2_fourier_hand_6dof_no_leg_no_collision.urdf"
                "ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/grx_description/GR1T2/urdf/GR1T2_fourier_hand_6dof_no_leg.urdf"
    )

    # 获取机器人所有的 Link
    for link in robot.get_links():
        for visual in link.get_visual_bodies():
            for render_shape in visual.get_render_shapes():
                material = render_shape.material
                material.base_color = np.array([1, 0, 0, 1])  # 变成红色
    for link in robot.get_links():
        print(link.get_name())
    print(robot.get_links())
    robot_joints = robot.get_active_joints()
    for robot_joint in robot_joints:
        print(robot_joint.get_name())
    
    for joint_idx, joint in enumerate(robot.get_active_joints()):
        joint.set_drive_property(stiffness=1e5, damping=1e3, force_limit=300,mode='force')
    
    time.sleep(5)

    robot.set_root_pose(sapien.Pose([0, 0, 0.98], [1, 0, 0, 0]))
    # Set initial joint positions
    qpos = [ 
        0, 0, 0, 0, 0, 0, 0, 
        # 0, 1.22, 1.22, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.74, 0, 0,
        -1.57, -1.74,
        -1.57, -1.74,
        -1.57, -1.74,
        -1.57, -1.74,
    ]
    robot.set_qpos(qpos)
    print(robot.get_qpos())
    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=1e5, damping=1e3)

    camera = scene.add_camera(
        name="camera",
        width=int(848),
        height=int(480),
        fovy=np.deg2rad(78.0),  # D435 fovy
        near=0.1,
        far=10.0,
    )
    camera.set_focal_lengths(605.12, 604.91)
    camera.set_principal_point(424.59, 236.67)
    link_camera = [x for x in robot.get_links() if x.name == "head_yaw_link"][0]
    # camera.set_parent(parent=link_camera, keep_pose=False)
    # camera.set_local_pose(
    #     sapien.Pose.from_transformation_matrix(np.array([[np.cos(math.pi/6), 0, np.sin(math.pi/6), 0], [0, 1, 0, 0], [-np.sin(math.pi/6),0,np.cos(math.pi/6), 0], [0, 0, 0, 1]]))
    #     # sapien.Pose.from_transformation_matrix(np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
    # )  # SAPIEN uses ros camera convention; the rotation matrix of link_camera's pose is in opencv convention, so we need to transform it to ros convention
        # ROS相机规范：ROS 中的相机坐标系通常定义为：
        # x 轴朝向相机的前方
        # y 轴朝左
        # z 轴朝上

        # OpenCV相机规范：OpenCV 的相机坐标系定义为：
        # x 轴朝右
        # y 轴朝下
        # z 轴朝前方

    tcp_link = [x for x in robot.get_links() if x.name == "left_end_effector_link"][0]
    i =0
    while not viewer.closed:
        # print(robot.get_qpos())
        star = 0
        if math.sin(0.01*i) > 0:
            star = math.sin(0.01*i)
    #     qpos = [ 
    #     0. ,0, 0, -math.pi/2, 0, 0, 0, 
    #     -1.74*star, 
    #     -1.57*star,
    #     -1.57*star,
    #     -1.57*star,
    #     -1.57*star,
    #     0, 
    #     -1.74*star,
    #     -1.74*star,
    #     -1.74*star,
    #     -1.74*star,
    #     0, 
    # ]
        qpos = [    
        0. ,0, 0, -math.pi/2, 0, 0, 0, 
        -1.74*star,
        1.22*star,
        1.22*star,
        -1.57*star,
        -1.74*star,
        -1.57*star,
        -1.74*star,
        -1.57*star,
        -1.74*star,
        -1.57*star,
        -1.74*star,
    ]
        i+=1
        for joint_idx, joint in enumerate(robot.get_active_joints()):
                joint.set_drive_target(qpos[joint_idx])
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                    # external=False
                )
                print("qf:-------------",qf)
                robot.set_qf(qf)
            # print("target qpos", qpos)
            # print("current qpos", robot.get_qpos())
            # print("tcp pose wrt robot base", robot.pose.inv() * tcp_link.pose)
            # robot.set_qpos(qpos)
            # robot.set_drive_target(qpos)
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    demo(fix_root_link=True, balance_passive_force=True)


if __name__ == "__main__":
    main()