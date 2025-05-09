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

from mmdemo.utils.cliff_utils.common.depth_operations import px_to_cam, get_pelvis_translation, get_probe_centroid, depth_scaled_metric, smpl_fix_coordinates
from mmdemo.utils.cliff_utils.common.scene_operations import rgb_hsv_mask, centroid_px, add_reference_frame, add_prob_centroid2scene
from mmdemo.utils.cliff_utils.common.smpl_fitting_ops import refine_smpl, smpl_skip_refinement
from mmdemo.utils.cliff_utils.common.preprocessing_operations import map_kinect_to_smpl, process_keypoints, get_crop_cam, preprocess_crop, compute_bbox_full_scale
from mmdemo.utils.cliff_utils.losses import camera_fitting_loss, body_fitting_loss
from mmdemo.utils.cliff_utils.prior import MaxMixturePrior


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
        patientConfidence = []
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
                    patientConfidence.append(joint[3] / 2.0) #convert to 0, 0.5 ot 1.0
                if(bodyId == 2):
                    practitioner.append(point)


        if len(patient) == 0:
            return SceneInterface(mesh_scene=None, smpl_joints=None, probe_centroid=None, pelvis_coords=None)
        patient_azure_keypoints = np.array(patient).reshape(32,2)
        patient_azure_confidence = np.array(patientConfidence).reshape(32,1)
        
        # getting RGB image and depth images
        frame = color.frame
        frame = frame[:, :, ::-1] # need to convert from RGB to BGR for CLIFF
        depth_frame = depth.frame
        depth_map = depth_frame/1000.0


        # Camera Calibration

        K = calibration.camera_matrix
        focal_length_value = (K[0,0] + K[1,1]) / 2.0
        camera_center = np.array([960, 540])

        # MAP from Kinect to openpose sequence of joints
        keypoints = process_keypoints(patient_azure_keypoints, patient_azure_confidence)

        # Get translation for SMPL
        try:
            pelvis_translation = get_pelvis_translation(patient_azure_keypoints[0],depth_map, K, self.device)
        except:
            pelvis_translation = None
        
        img_w=1920.0
        img_h=1080.0
        bbox, w, h = compute_bbox_full_scale(patient_azure_keypoints, img_w, img_h)


        img_h_t = torch.tensor([img_h], dtype=torch.float32, device=self.device)
        img_w_t = torch.tensor([img_w], dtype=torch.float32, device=self.device)
        full_img_shape = torch.stack((img_h_t, img_w_t), dim=-1) 
        
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
            pred_rotmat, betas, pred_cam_crop = self.cliff_model(norm_img, bbox_info, n_iter=5)

        # Use data_full_cam if using translation data from Kinect
        data_full_cam = torch.tensor([[kinect_translation[0]/1000,  -kinect_translation[1]/1000, kinect_translation[2]/1000]], dtype=torch.float32, device=self.device)
        
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
            num_iters=1,
            device=self.device)

        try:
            new_opt_vertices, new_joints = smpl_fix_coordinates(new_opt_vertices,new_opt_joints, pelvis_translation)
        except:
            new_opt_vertices = new_opt_vertices
            new_joints = new_opt_joints
            
        vertices = new_opt_vertices.cpu().detach().numpy()
        if vertices.ndim == 3:
            vertices = vertices[0]
        if not isinstance(faces, np.ndarray):
            faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces
        
        
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        smpl_joints = new_joints.cpu().detach().numpy()[0]  # shape (num_joints, 3)

        return SceneInterface(mesh_scene=body_mesh, smpl_joints=smpl_joints, probe_centroid=centroid_3d, pelvis_coords=pelvis_translation) 
