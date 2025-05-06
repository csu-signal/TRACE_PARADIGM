from threading import Lock
from queue import SimpleQueue
import warnings
from pathlib import Path
from typing import final
import time
import joblib
import mediapipe as mp
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import get_average_hand_pixel, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    LandmarkInterface,
    SceneInterface,
)
from mmdemo.interfaces.data import Cone, Handedness, Landmarks
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d

import os
import json
import argparse
import numpy as np
import torch
import trimesh
import pyrender
from pytorch3d import transforms
import cv2 as cv

from mmdemo.utils.cliff_utils.common import constants
from mmdemo.utils.cliff_utils.common.utils import strip_prefix_if_present, cam_crop2full, full2crop_cam
from mmdemo.utils.cliff_utils.common.constants import SMPL_CKPT_HR48, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, SMPL_CKPT_RES50
from mmdemo.utils.cliff_utils.models.smpl import SMPL    
from mmdemo.utils.cliff_utils.smplify import SMPLify  
from mmdemo.utils.cliff_utils.losses import perspective_projection
from mmdemo.utils.cliff_utils.models.cliff_hr48.cliff import CLIFF as cliff_hr48
from mmdemo.utils.cliff_utils.models.cliff_res50.cliff import CLIFF as cliff_res50 
from mmdemo.utils.cliff_utils.common.imutils import process_image

from mmdemo.utils.cliff_utils.common.depth_operations import px_to_cam, get_pelvis_translation, get_probe_centroid, depth_scaled_metric
from mmdemo.utils.cliff_utils.common.scene_operations import rgb_hsv_mask, centroid_px, add_reference_frame, add_prob_centroid2scene
from mmdemo.utils.cliff_utils.common.smpl_fitting_ops import refine_smpl, smpl_skip_refinement
from mmdemo.utils.cliff_utils.common.preprocessing_operations import map_kinect_to_smpl, process_keypoints, get_crop_cam, preprocess_crop, compute_bbox_full_scale
from mmdemo.utils.cliff_utils.losses import camera_fitting_loss, body_fitting_loss
from mmdemo.utils.cliff_utils.prior import MaxMixturePrior

# SMPL expected joint ordering as provided
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Kinect joint ordering (indices) and their names are defined as:
#   0: PELVIS
#   1: SPINE_NAVEL
#   2: SPINE_CHEST
#   3: NECK
#   4: CLAVICLE_LEFT
#   5: SHOULDER_LEFT
#   6: ELBOW_LEFT
#   7: WRIST_LEFT
#   8: HAND_LEFT
#   9: HANDTIP_LEFT
#   10: THUMB_LEFT
#   11: CLAVICLE_RIGHT
#   12: SHOULDER_RIGHT
#   13: ELBOW_RIGHT
#   14: WRIST_RIGHT
#   15: HAND_RIGHT
#   16: HANDTIP_RIGHT
#   17: THUMB_RIGHT
#   18: HIP_LEFT
#   19: KNEE_LEFT
#   20: ANKLE_LEFT
#   21: FOOT_LEFT
#   22: HIP_RIGHT
#   23: KNEE_RIGHT
#   24: ANKLE_RIGHT
#   25: FOOT_RIGHT
#   26: HEAD
#   27: NOSE
#   28: EYE_LEFT
#   29: EAR_LEFT
#   30: EYE_RIGHT
#   31: EAR_RIGHT

# Create a mapping dictionary from the SMPL joint names to the Kinect indices.
# For joints not directly available (or needing a combination), we provide special instructions.
joint_mapping = {
    # OpenPose joints
    'OP Nose': 27,
    'OP Neck': 3,
    'OP RShoulder': 12,
    'OP RElbow': 13,
    'OP RWrist': 14,
    'OP LShoulder': 5,
    'OP LElbow': 6,
    'OP LWrist': 7,
    # For OP MidHip, use the average of the left and right hips (indices 18 and 22)
    'OP MidHip': 0,
    'OP RHip': 22,
    'OP RKnee': 23,
    'OP RAnkle': 24,
    'OP LHip': 18,
    'OP LKnee': 19,
    'OP LAnkle': 20,
    'OP REye': 30,   # Kinect's EYE_RIGHT
    'OP LEye': 28,   # Kinect's EYE_LEFT
    'OP REar': 31,   # Kinect's EAR_RIGHT
    'OP LEar': 29,   # Kinect's EAR_LEFT
    # The following joints are not provided by Kinect.
    'OP LBigToe': 21,
    'OP LSmallToe': None,
    'OP LHeel': None,
    'OP RBigToe': 25,
    'OP RSmallToe': None,
    'OP RHeel': None,
    # Ground Truth joints
    'Right Ankle': 24,
    'Right Knee': 23,
    'Right Hip': 22,
    'Left Hip': 18,
    'Left Knee': 19,
    'Left Ankle': 20,
    'Right Wrist': 14,
    'Right Elbow': 13,
    'Right Shoulder': 12,
    'Left Shoulder': 5,
    'Left Elbow': 6,
    'Left Wrist': 7,
    'Neck (LSP)': 3,
    'Top of Head (LSP)': 26,
    'Pelvis (MPII)': 0,
    'Thorax (MPII)': 2,  # Using SPINE_CHEST
    'Spine (H36M)': 1,   # Using SPINE_NAVEL
    'Jaw (H36M)': None,  # Not provided by Kinect
    'Head (H36M)': 26,
    'Nose': 27,
    'Left Eye': 28,
    'Right Eye': 30,
    'Left Ear': 29,
    'Right Ear': 31
}

@final
class CliffPose(BaseFeature[SceneInterface]):
    """
    Convert Pose Data to CLIFF visualization.

    Input interfaces are `ColorImageInterface`, `DepthImageInterface`,
    `BodyTrackingInterface`, `CameraCalibrationInterface`

    Output interface is `SceneInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        bt: BaseFeature[BodyTrackingInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
    ):
        super().__init__(color, depth, bt, calibration)

    def initialize(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
         # Load the pretrained CLIFF model.
        # cliff = eval("cliff_res50")
        cliff = eval("cliff_hr48")
        self.cliff_model = cliff(SMPL_MEAN_PARAMS).to(self.device)
        # state_dict = torch.load(SMPL_CKPT_RES50)['model']
        state_dict = torch.load(SMPL_CKPT_HR48)['model']
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        self.cliff_model.load_state_dict(state_dict, strict=True)
        self.cliff_model = self.cliff_model.to(self.device)
        self.cliff_model.eval()

        self.smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(self.device)
        self.smplify = None

        self.bbox_info = None

        self.pose_prior = MaxMixturePrior(prior_folder='data',num_gaussians=8,dtype=torch.float32).to(self.device)


    def get_output(
        self,
        color: ColorImageInterface,
        depth: DepthImageInterface,
        bt: BodyTrackingInterface,
        calibration: CameraCalibrationInterface,
    ):
        if not color.is_new() or not depth.is_new() or not bt.is_new() or not calibration.is_new():
            return None

        
        

        # get body tracking info (azure_keypoints)
        practitioner = []
        patient = []
        bt = fix_body_id(bt)
        for bodyIndex, body in enumerate(bt.bodies):  
            bodyId = int(body["wtd_body_id"])
            for jointIndex, joint in enumerate(body["joint_positions"]):
                points2D, _ = cv.projectPoints(
                    np.array(joint[:3]), 
                    calibration.rotation,
                    calibration.translation,
                    calibration.camera_matrix,
                    calibration.distortion) 
                point = (int(points2D[0][0][0]),int(points2D[0][0][1]))  
                if(bodyId == 1):
                    patient.append(point)
                if(bodyId == 2):
                    practitioner.append(point)


        if len(patient) == 0:
            return None
        patient_azure_keypoints = np.array(patient).reshape(32,2)
        
        # getting RGB image and depth images
        frame = color.frame
        depth_frame = depth.frame
        # print(depth_frame.shape)
        # np.save("./original-depth-frame.npz", depth_frame)
        depth_image_8bit = cv.normalize(depth_frame, None, 0, 255, )
        # np.save("./normalized-depth-frame.npz", depth_image_8bit)

        # depth_frame = cv.applyColorMap(depth_image_8bit, cv.IMREAD_GRAYSCALE)
        depth_map = depth_scaled_metric(depth_image_8bit, near = 0.5, far = 5.5, offset = 0.3)
        # np.save("./resulting-depth-map.npz", depth_map)
        # print(depth_image_8bit.shape, depth_map.shape,)


        # Camera Calibration

        K = calibration.camera_matrix
        focal_length_value = (K[0,0] + K[1,1]) / 2.0
        camera_center = np.array([960, 540])

        # MAP from Kinect to openpose sequence of joints
        keypoints = process_keypoints(patient_azure_keypoints)

        # Get translation for SMPL
        # try:
        pelvis_translation = get_pelvis_translation(patient_azure_keypoints[0],depth_map, K, self.device)
        # except Exception as e:
        #     print(e)
        #     pelvis_translation = None
        # Compute bounding box
        img_w=1920.0
        img_h=1080.0
        bbox, w, h = compute_bbox_full_scale(patient_azure_keypoints, img_w, img_h)


        img_h_t = torch.tensor([img_h], dtype=torch.float32, device=self.device)
        img_w_t = torch.tensor([img_w], dtype=torch.float32, device=self.device)
        full_img_shape = torch.stack((img_h_t, img_w_t), dim=-1) 
        
        # norm_img = self.preprocess_frame(frame, (224, 224)).to(self.device)
        focal_length = torch.tensor([focal_length_value], dtype=torch.float32, device=self.device) #500
        camera_center_tensor = torch.tensor(np.array(camera_center), dtype=torch.float32, device=self.device)
        

        # Get 3D coordinates of probe centroid
        try:
            centroid_3d = get_probe_centroid(frame, depth_map, K)        
        except:
            centroid_3d = None
        # Get proprocessing data for CLIFF 
        # For debuggin visualize crop_img to check the image fed to CLIFF
        norm_img, center, scale, ul, br, crop_img, bbox_info, b = preprocess_crop(frame, bbox, crop_height=224, crop_width=224, camera_center=camera_center, focal_length=focal_length, device=self.device)

        ### Pass the Kinect translation to get_crop_cam to get the camera initialization
        kinect_translation = calibration.translation
        init_cam = get_crop_cam(kinect_translation, center, b, camera_center_tensor, focal_length, device=self.device)

        # Run CLIFF
        with torch.no_grad():
            pred_rotmat, betas, pred_cam_crop = self.cliff_model(norm_img, bbox_info,init_cam=init_cam, n_iter=5)

        # Use data_full_cam if using translation data from Kinect
        data_full_cam = torch.tensor([[kinect_translation[0]/1000,  -kinect_translation[1]/1000, kinect_translation[2]]], dtype=torch.float32, device=self.device)
        
        # Process pose data
        init_pose = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)    
        
        # For reference - >refine_smpl(smpl,betas, init_pose, pelvis_translation,pred_cam_full, keypoints, camera_center_tensor, focal_length,pose_prior, num_iters=5, device="cuda")
        new_opt_vertices, new_opt_joints, new_opt_pose,new_opt_betas, faces = refine_smpl(smpl=self.smpl, betas=betas,
            init_pose= init_pose, 
            pelvis_translation =pelvis_translation,
            pred_cam_full =data_full_cam,
            keypoints= keypoints,
            camera_center_tensor =camera_center_tensor,
            focal_length = focal_length,
            pose_prior=self.pose_prior,
            num_iters=5,
            device=self.device)

        vertices = new_opt_vertices.cpu().detach().numpy()
        if vertices.ndim == 3:
            vertices = vertices[0]
        if not isinstance(faces, np.ndarray):
            faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces
        
        
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        smpl_joints = new_opt_joints.cpu().detach().numpy()[0]  # shape (num_joints, 3)

        return SceneInterface(mesh_scene=body_mesh, smpl_joints=smpl_joints, probe_centroid=centroid_3d) 
    

    def map_kinect_to_smpl(self, kinect_keypoints):
        """
        Map Kinect keypoints to SMPL joint ordering.
        
        Parameters:
        kinect_keypoints (np.array): Array of shape (32, D) where D is the dimensionality of each joint (e.g. 2 for 2D, 3 for 3D).
        
        Returns:
        np.array: An array of shape (49, D) arranged in the SMPL joint order.
                For joints not available from Kinect, a zero vector is inserted.
        """
        # Determine the dimensionality (e.g., 2D or 3D) from the Kinect keypoints.
        num_dims = kinect_keypoints.shape[1] if len(kinect_keypoints.shape) > 1 else 1
        smpl_keypoints = []
        
        for joint_name in JOINT_NAMES:
            mapping = joint_mapping[joint_name]
            if mapping is None:
                # Joint not available: fill with zeros.
                smpl_keypoints.append(np.zeros(num_dims))
            else:
                smpl_keypoints.append(kinect_keypoints[mapping])
        
        return np.array(smpl_keypoints)

    def preprocess_frame(self, frame, target_size=(224, 224)):
        """
        Preprocess the input frame (BGR) to a normalized tensor.
        Returns norm_img as a tensor of shape (1, 3, H, W) in float32.
        """
        resized = cv.resize(frame, target_size)
        rgb = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5  
        #rgb = resized/255.0 #cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        #cv2.imwrite("norm_img.jpg", rgb*255)
        norm_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        return norm_img