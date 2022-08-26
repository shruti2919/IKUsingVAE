# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jdj0ayShAgZkpklrYuFylY_C2bva_Cy2
"""

# pytorch
import torch
from torch import nn

# python
from typing import Optional, Tuple

CVAE_DEFAULT_CONFIG = {
    # 6 if euler angles, 7 if quaternion
    "pose_dims": 6,
    # number of actuated joints 
    "joint_dims": 7,
    # dimension of hidden layer
    "hidden_dims": 128,
    # dimension of latent space
    "latent_dims": 1,
    # lower joint position limits
    "lower_joint_limits": torch.Tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    # upper joint position limits
    "upper_joint_limits": torch.Tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]),
}


class ConstrainedCVAE(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # updated config
        self._config = CVAE_DEFAULT_CONFIG
        print(CVAE_DEFAULT_CONFIG)
        if config is not None:
            self._config.update(config)
        # joint position range:
        self._joint_position_range = (self._config["upper_joint_limits"] - self._config["lower_joint_limits"]).to(self.device)
        # joint position mean:
        self._joint_position_mean = 0.5*(self._config["upper_joint_limits"] + self._config["lower_joint_limits"]).to(self.device)

        # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(self._config["pose_dims"] + self._config["joint_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], 2*self._config["latent_dims"])
        )
        # decoder takes latent space + desired pose
        self.decoder = nn.Sequential(
            nn.Linear(self._config["latent_dims"]+self._config["pose_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            #nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            #nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["joint_dims"]),
            nn.Tanh()
        )

    def forward(self, desired_pose: torch.Tensor, joint_config: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        if self.training:
            # forward through encoder to get distribution params
            latent_params = self.encoder(torch.cat((joint_config, desired_pose), axis=1).view(-1, self._config["pose_dims"] + self._config["joint_dims"]))
            # sample mean
            mean = latent_params[:, 0:self._config["latent_dims"]]
            # convert log of variance to standard deviation
            log_variance = latent_params[:, self._config["latent_dims"]:]
            stddev = log_variance.mul(0.5).exp_()
            # noise sample from standard normal distribution
            std_norm_var = stddev.new_empty(stddev.size()).normal_()
            # reparameterization trick to sample from distribution
            z = std_norm_var.mul_(stddev).add_(mean)
            # run through decoder
            output = self.decoder(torch.cat((z, desired_pose), axis=1))
            # scale the output to meet joint position constraints
            q = output*0.5*self._joint_position_range + self._joint_position_mean
            return q, mean, log_variance
        else:
            # if z is not provided, sample from standard normal
            if z == None:
                z = torch.empty((desired_pose.size()[0], self._config["latent_dims"])).normal_(mean=0, std=1)
            z = z.to(self.device)
            # run through decoder
            print(z.shape, desired_pose.shape)
            output = self.decoder(torch.cat((z, desired_pose), axis=1))
            # scale the output to meet joint position constraints
            q = output*0.5*self._joint_position_range + self._joint_position_mean
            return q

# pytorch
import torch
from torch.utils.data import Dataset

# data generating config
#from data import DataGenConfig
# from data_config import DataGenConfig

# python
import numpy as np

class IKDataset(Dataset):
    def __init__(self):
        dataset = np.loadtxt(training_targets_path, 
                        delimiter=",",
                        dtype = np.float32,
                        skiprows=1)
        # dataset = np.loadtxt(DataGenConfig.OUT_FILE_NAME, 
        #                 delimiter=",",
        #                 dtype = np.float32,
        #                 skiprows=1)
        pose = np.array([data[0:7] for data in dataset])
        configuration = np.array([data[7:] for data in dataset])

        # converting to torch tensors
        self.x = torch.from_numpy(pose)
        self.y = torch.from_numpy(configuration)
        self.n_samples = dataset.shape[0]
        print(self.x.shape)
    
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# for testing purposes
if __name__ == "__main__":
    dataset = IKDataset()
    pose, configuration = dataset[0]
    print(pose, configuration)

## Mounting 
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd

root_dir = '/content/drive/MyDrive/' # this is to be defined by you 
local_path = '/CW1/' # store the related data files in this folder

data_dir = root_dir + local_path
## Define paths to the training data and targets files

training_targets_path = data_dir + 'franka_ik_data_final.txt'
urdf_path  = data_dir + 'panda_arm.urdf'
expert_cartesian_poses_path = data_dir+'expert_cartesian_poses.npy'
constrained_cvae_weights_pth = data_dir + 'constrained_cvae_weights_final.pth'

# pytorch
import torch
from torch.utils.data import Dataset

# data generating config
#from data import DataGenConfig
# from data_config import DataGenConfig

# python
import numpy as np

class IKDataset(Dataset):
    def __init__(self):
        dataset = np.loadtxt(training_targets_path, 
                        delimiter=",",
                        dtype = np.float32,
                        skiprows=1)
        # dataset = np.loadtxt(DataGenConfig.OUT_FILE_NAME, 
        #                 delimiter=",",
        #                 dtype = np.float32,
        #                 skiprows=1)
        pose = np.array([data[0:7] for data in dataset])
        configuration = np.array([data[7:] for data in dataset])

        # converting to torch tensors
        self.x = torch.from_numpy(pose)
        self.y = torch.from_numpy(configuration)
        self.n_samples = dataset.shape[0]
        print(self.x.shape)
    
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# for testing purposes
if __name__ == "__main__":
    dataset = IKDataset()
    pose, configuration = dataset[0]
    print(pose, configuration)

pip install differentiable-robot-model

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pip install pin

#import pytorch
import pinocchio
#from data.data_config import DataGenConfig

import torch
from torch import nn
import numpy as np
from os.path import dirname, abspath, join
from differentiable_robot_model.robot_model import DifferentiableRobotModel
#from model import ConstrainedCVAE




def inference(pose, z = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = ConstrainedCVAE().to(device)
    cvae.load_state_dict(torch.load(constrained_cvae_weights_pth, map_location=device))
    cvae.eval()

    with torch.no_grad():
        pose = pose.to(device)
        q = cvae(desired_pose = pose, z = z)
        return q



if __name__ == "__main__":
    cat_trajs = np.load(expert_cartesian_poses_path, allow_pickle=True)
    cat_trajs = np.vstack(cat_trajs)
    x=0
    for pose in cat_trajs:
        pose = torch.Tensor([pose])
        x = x+1
    # pose = torch.Tensor([-0.3622564536646905,0.07453657615711093,0.523455111826844,0.6949510146762841,0.6371909076037253,-0.28704989069534576,-0.16921345903719948])
    #pose = torch.Tensor([[0.55641491, -0.04412349, 0.18583907, -0.33566937, 0.92741494, -0.13821891, 0.09012842]])#([[0.5, 0, 0.5, 0, 0, 0, 1]])
        z = None
        #z = torch.Tensor([0, 0, 0])
        q = inference(pose=pose, z=z)[0]

        #print("Generated q: ", q)


        #pinocchio_model_dir = dirname(dirname(str(abspath(__file__)))) 
        #model_path = pinocchio_model_dir + "/learning-ik/resources/" + DataGenConfig.ROBOT
        #urdf_path = model_path + "/urdf/"+DataGenConfig.ROBOT_URDF
        # setup robot model and data
        #urdf_path = "C:/Masters project/Franka_Panda_IK_Sensor-main/Franka_Panda_IK_Sensor-main/learning-ik/resources/franka/urdf/panda_arm.urdf"
        model = pinocchio.buildModelFromUrdf(urdf_path)
        #model = DifferentiableRobotModel(urdf_path, name="franka_panda")
        data = model.createData()
        # setup end effector
        ee_name = "panda_link7"
        ee_link_id = model.getFrameId(ee_name)
        # joint limits (from urdf)
        lower_limit = np.array(model.lowerPositionLimit)
        upper_limit = np.array(model.upperPositionLimit)
        pinocchio.framesForwardKinematics(model, data, q.cpu().numpy())
        desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])

        #print("Desired Pose", pose[:].cpu().numpy())
        #print("Pose: {} ||  Error: {} ".format(x,  np.linalg.norm(pose[:].cpu().numpy() - desired_pose[:])))
        print("Desired Pose", pose[:].cpu().numpy())
        print("Generated Pose: ", desired_pose[:])
        print("Error: ", np.linalg.norm(pose[:].cpu().numpy() - desired_pose[:]))
        





    print("------------------------------------------------------\n")

    for i in range (10):
        z = torch.Tensor([2*i-1])
        q = inference(pose=pose, z=z)
        pinocchio.framesForwardKinematics(model, data, q.cpu().numpy())
        desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        #print(list(q.cpu().numpy()))
        #print("Pose",i)
        #print("Error: ", np.linalg.norm(pose[:3].cpu().numpy() - desired_pose[:3]))
        print("\n")

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# differentiable-robot-model
from differentiable_robot_model.robot_model import DifferentiableRobotModel

#import ConstrainedCVAE
#import IKDataset 

# training hyperparameters
BATCH_SIZE = 25
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
MOMENTUM = 0.95
LEARNING_RATE_DECAY = 0.97
#LEARNING_RATE_DECAY = 3e-6

def cvae_fk_loss(joint_config: torch.Tensor, true_pose: torch.Tensor,
                 mean: torch.Tensor, log_variance: torch.Tensor, 
                 robot_model: DifferentiableRobotModel, beta: float = 0.02) -> torch.Tensor:
    pose = torch.cat(robot_model.compute_forward_kinematics(joint_config, "panda_link7"), axis=1)
    # reconstruction loss in task space
    recon_loss = nn.functional.mse_loss(true_pose, pose, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss


def cvae_loss(joint_config: torch.Tensor, true_joint_config: torch.Tensor, 
              mean: torch.Tensor, log_variance: torch.Tensor, 
              beta: float = 0.01) -> torch.Tensor:
    # reconstruction loss in configuration space
    recon_loss = nn.functional.mse_loss(joint_config, true_joint_config, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss

def train():
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = ConstrainedCVAE().to(device)
    cvae.train()

    # generate dataset
    print("--------------Loading Data--------------")
    dataset = IKDataset()
    # shuffle=False because each data point is already randomly sampled
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("--------------Data  Loaded--------------\n")

    # optimizer
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    beta = 0.02

    # setup differentiable robot model stuff
   # urdf_path = "C:/Masters project/Franka_Panda_IK_Sensor-main/Franka_Panda_IK_Sensor-main/learning-ik/resources/franka/urdf/panda_arm.urdf"
    robot_model = DifferentiableRobotModel(
        urdf_path, name="franka_panda", device=str(device)
    )

    # training loop
    print("----------------Training----------------")
    for epoch in range(1):
        epoch_error = 0
        for pose, joint_config in train_loader:
            #print("pose : {}".format(pose))
            #print("config: {}".format(joint_config))
            pose = pose.to(device)
            joint_config = joint_config.to(device)
            print("pose : {}".format(pose))
            print("config: {}".format(joint_config))
            joint_config_pred, mean, log_variance = cvae(pose, joint_config)
            # loss = cvae_loss(joint_config_pred, joint_config,
            #                    mean, log_variance, beta)
            loss = cvae_fk_loss(joint_config_pred, pose, 
                                mean, log_variance, robot_model, beta)
            epoch_error += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break 
        if epoch > 25:
            optimizer_scheduler.step()
        print("Epoch Number: {} || Average Error: {}  || loss: {}".format(epoch, epoch_error/dataset.n_samples,loss))
    print("-----------Training  Completed-----------")

    # save the weights
    torch.save(cvae.state_dict(), result_path)
    

if __name__ == "__main__":
    train()



#import pytorch
import pinocchio
#from data.data_config import DataGenConfig

import torch
from torch import nn
import numpy as np
from os.path import dirname, abspath, join
from differentiable_robot_model.robot_model import DifferentiableRobotModel
#from model import ConstrainedCVAE

#pose = pose.to(device)
#joint_config = joint_config.to(device)


def inference(pose, z = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = ConstrainedCVAE().to(device)
    cvae.load_state_dict(torch.load(constrained_cvae_weights_pth, map_location=device))
    cvae.eval()

    with torch.no_grad():
        pose = pose.to(device)
        q = cvae(desired_pose = pose, z = z)
        return q



if __name__ == "__main__":
  z = None
  pose = [0, 0, 0, -0.33566937, 0.92741494, -0.13821891, 0.09012842]
  pose = torch.Tensor([pose])
  q = inference(pose=pose, z=z)[0]
  model = pinocchio.buildModelFromUrdf(urdf_path)
  data = model.createData()
  #data = model.createData()
  # setup end effector
  ee_name = "panda_link7"
  ee_link_id = model.getFrameId(ee_name)
        # joint limits (from urdf)
  lower_limit = np.array(model.lowerPositionLimit)
  upper_limit = np.array(model.upperPositionLimit)
  pinocchio.framesForwardKinematics(model, data, q.cpu().numpy())
  desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
  print("Desired Pose", pose[:].cpu().numpy())
  print("Generated Pose: ", desired_pose[:])
  print("Error: ", np.linalg.norm(pose[:].cpu().numpy() - desired_pose[:]))

result_path  = data_dir + 'constrained_cvae_weights.pth'

import matplotlib.pyplot as plt

plt.plot(cvae.cvae['accuracy'])
plt.plot(cvae.cvae['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()