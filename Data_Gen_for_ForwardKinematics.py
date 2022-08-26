import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R
from os.path import dirname, abspath, join

urdf_path = 'C:/Masters project/Franka_Panda_IK_Sensor-main/Franka_Panda_IK_Sensor-main/learning-ik/resources/franka/urdf/kuka_arm.urdf'


def gen_rand_config(lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
    return np.random.uniform(low=lower_limit, high=upper_limit)

def gen_rand_exp_config(lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
    jmax = [ 0.46026527, 0.66581495, -0.3606656, -1.87172568, 0.06517799, 2.99494907]
    jmin = [ 0.30369494, 0.48949512, -0.51267991, -2.27541177,  0.06016369,  2.4149924]
    return np.random.uniform(low=jmin, high=jmax)

def generate_data():
    model = pinocchio.buildModelFromUrdf(urdf_path)
    lower_limit = np.array(model.lowerPositionLimit)
    upper_limit = np.array(model.upperPositionLimit)
    data = model.createData()
    # setup end effector
    ee_name = 'link_6'
    ee_link_id = model.getFrameId(ee_name)
    # joint limits (from urdf)
    th = gen_rand_config(lower_limit, upper_limit)
    file_name = "kuka_ik_data_1000.txt"
    file = open(file_name, "w")
    file.write("Pose\tConfiguration\n")
    num_data = 1000
    IS_QUAT = True
    JOINT_DIMS =7
    # data generating loop
    for i in range(num_data):  # num_data):
        # generating feature and label
        config = gen_rand_config(lower_limit, upper_limit)
        pinocchio.framesForwardKinematics(model, data, config)
        pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        # converting quaternion to euler angle
        if not IS_QUAT:
            rotation = R.from_quat(list(pose[3:]))
            rotation_euler = rotation.as_euler("xyz")
            pose = np.concatenate((pose[0:3], rotation_euler))
        # annoying string manipulation for saving in text file
        # if we only care about a subset of the total chain
        config = config[:JOINT_DIMS]
        str_pose = [str(i) for i in pose]
        str_config = [str(i) for i in config]
        file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")
        str_pose = [str(i) for i in pose]
        str_config = [str(i) for i in config]
        file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")

    file.close()

if __name__ == "__main__":
    generate_data()
