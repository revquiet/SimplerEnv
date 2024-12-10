import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_quaternion_to_euler(quat):
    """
    Convert Quarternion (xyzw) to Euler angles (rpy) 
    """
    # Normalize
    quat = quat / np.linalg.norm(quat)
    euler = R.from_quat(quat).as_euler('xyz')

    return euler


def convert_euler_to_quaternion(euler):
    """
    Convert Euler angles (rpy) to Quarternion (xyzw)
    """
    quat = R.from_euler('xyz', euler).as_quat()
    
    return quat


def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()
    
    return quat


def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=True)
    
    return euler

def convert_quaternion_to_rotation_matrix(quat):
    """
    Convert Quarternion (xyzw) to rotation matrix (3x3).
    """
    if np.allclose(quat, 0):
        return np.zeros((3, 3))
    rotmat = R.from_quat(quat).as_matrix()

    return rotmat

def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out
        

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix


def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    # ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    ortho6d = matrix[:, :2].transpose(1, 0).reshape(-1)
    return ortho6d

def conver_ortho6d_from_quaternion(quaternion):
    rotmat = convert_quaternion_to_rotation_matrix(quaternion)
    ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)
    return ortho6d

def conver_ortho6d_to_euler(ortho6d, squeeze=True):
    if ortho6d.ndim == 1:
        ortho6d = np.expand_dims(ortho6d, axis=0)
    rotation = compute_rotation_matrix_from_ortho6d(ortho6d)
    euler = convert_rotation_matrix_to_euler(rotation)
    if squeeze:
        euler = np.squeeze(euler)
    return euler

def conver_qpos_to_6drot(qpos_action):
    quat_1 = qpos_action[3:7]
    quat_2 = qpos_action[-4:]
    ortho6d_1 = conver_ortho6d_from_quaternion(quat_1)
    ortho6d_2 = conver_ortho6d_from_quaternion(quat_2)
    
    ortho6d_action = np.concatenate((qpos_action[:3], ortho6d_1, qpos_action[7:10], ortho6d_2), dtype=np.float32)
    return ortho6d_action

# Test
if __name__ == "__main__":
    # Randomly generate a euler ange
    euler = np.random.rand(3) * 2 * np.pi - np.pi
    euler = euler[None, :]    # Add batch dimension
    print(f"Input Euler angles: {euler}")
    
    # Convert to 6D Rotation
    rotmat = convert_euler_to_rotation_matrix(euler)
    print(f"Rotation matrix: {rotmat}")
    ortho6d = compute_ortho6d_from_rotation_matrix(rotmat)
    print(f"6D Rotation: {ortho6d}")
    print(f'shape: {ortho6d.shape}')
    
    # Convert back to Euler angles
    rotmat_recovered = compute_rotation_matrix_from_ortho6d(ortho6d)
    euler_recovered = convert_rotation_matrix_to_euler(rotmat_recovered)
    print(f"Recovered Euler angles: {euler_recovered}")
