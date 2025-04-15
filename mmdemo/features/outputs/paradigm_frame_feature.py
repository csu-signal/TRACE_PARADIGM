from typing import final
from enum import Enum, IntEnum
from xmlrpc.client import Boolean

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    GestureConesInterface,
    LandmarkInterface,
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import camera_3d_to_pixel


class HandJoints(IntEnum):
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP = 13,
    RING_FINGER_PIP = 14,
    RING_FINGER_DIP = 15,
    RING_FINGER_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20


class Joint(Enum):
        PELVIS = 0
        SPINE_NAVEL = 1
        SPINE_CHEST = 2
        NECK = 3
        CLAVICLE_LEFT = 4
        SHOULDER_LEFT = 5
        ELBOW_LEFT = 6
        WRIST_LEFT = 7
        HAND_LEFT = 8
        HANDTIP_LEFT = 9
        THUMB_LEFT = 10
        CLAVICLE_RIGHT = 11
        SHOULDER_RIGHT = 12
        ELBOW_RIGHT = 13
        WRIST_RIGHT = 14
        HAND_RIGHT = 15
        HANDTIP_RIGHT = 16
        THUMB_RIGHT = 17
        HIP_LEFT = 18
        KNEE_LEFT = 19
        ANKLE_LEFT = 20
        FOOT_LEFT = 21
        HIP_RIGHT = 22
        KNEE_RIGHT = 23
        ANKLE_RIGHT = 24
        FOOT_RIGHT = 25
        HEAD = 26
        NOSE = 27
        EYE_LEFT = 28
        EAR_LEFT = 29
        EYE_RIGHT = 30
        EAR_RIGHT = 31

class BodyCategory(Enum):
        HEAD = 0
        RIGHT_ARM = 1
        RIGHT_HAND = 7
        LEFT_ARM = 2
        LEFT_HAND = 6
        TORSO = 3
        RIGHT_LEG = 4
        LEFT_LEG = 5

def getPointSubcategory(joint):
     if(joint == Joint.PELVIS or joint == Joint.NECK or joint == Joint.SPINE_NAVEL or joint == Joint.SPINE_CHEST):
          return BodyCategory.TORSO
     if(joint == Joint.WRIST_LEFT or joint == Joint.CLAVICLE_LEFT or joint == Joint.SHOULDER_LEFT or joint == Joint.ELBOW_LEFT):
           return BodyCategory.LEFT_ARM
     if(joint == Joint.HAND_LEFT or joint == Joint.HANDTIP_LEFT or joint == Joint.THUMB_LEFT):
          return BodyCategory.LEFT_HAND
     if(joint == Joint.WRIST_RIGHT or joint == Joint.CLAVICLE_RIGHT or joint == Joint.SHOULDER_RIGHT or joint == Joint.ELBOW_RIGHT):
        return BodyCategory.RIGHT_ARM
     if(joint == Joint.HAND_RIGHT or joint == Joint.HANDTIP_RIGHT or joint == Joint.THUMB_RIGHT):
          return BodyCategory.RIGHT_HAND
     if(joint == Joint.HIP_LEFT or joint == Joint.KNEE_LEFT or joint == Joint.ANKLE_LEFT or joint == Joint.FOOT_LEFT):
          return BodyCategory.LEFT_LEG
     if(joint == Joint.HIP_RIGHT or joint == Joint.KNEE_RIGHT or joint == Joint.ANKLE_RIGHT or joint == Joint.FOOT_RIGHT):
          return BodyCategory.RIGHT_LEG
     if(joint == Joint.HEAD or joint == Joint.NOSE or joint == Joint.EYE_LEFT 
        or joint == Joint.EAR_LEFT or joint == Joint.EYE_RIGHT or joint == Joint.EAR_RIGHT):
          return BodyCategory.HEAD

hand_list = [
    [HandJoints.WRIST, HandJoints.INDEX_FINGER_MCP],
    [HandJoints.WRIST, HandJoints.THUMB_CMC],
    [HandJoints.WRIST, HandJoints.PINKY_MCP],
    [HandJoints.INDEX_FINGER_MCP, HandJoints.MIDDLE_FINGER_MCP],
    [HandJoints.INDEX_FINGER_MCP, HandJoints.INDEX_FINGER_PIP],
    [HandJoints.MIDDLE_FINGER_MCP, HandJoints.MIDDLE_FINGER_PIP],
    [HandJoints.MIDDLE_FINGER_MCP, HandJoints.RING_FINGER_MCP],
    [HandJoints.RING_FINGER_MCP, HandJoints.PINKY_MCP],
    [HandJoints.RING_FINGER_MCP, HandJoints.PINKY_PIP],
    [HandJoints.THUMB_CMC, HandJoints.THUMB_MCP],
    [HandJoints.THUMB_MCP, HandJoints.THUMB_IP],
    [HandJoints.THUMB_IP, HandJoints.THUMB_TIP],
    [HandJoints.INDEX_FINGER_PIP, HandJoints.INDEX_FINGER_DIP],
    [HandJoints.INDEX_FINGER_DIP, HandJoints.INDEX_FINGER_TIP],
    [HandJoints.MIDDLE_FINGER_PIP, HandJoints.MIDDLE_FINGER_DIP],
    [HandJoints.MIDDLE_FINGER_DIP, HandJoints.MIDDLE_FINGER_TIP],
    [HandJoints.RING_FINGER_PIP, HandJoints.RING_FINGER_DIP],
    [HandJoints.RING_FINGER_DIP, HandJoints.RING_FINGER_TIP],
    [HandJoints.PINKY_PIP, HandJoints.PINKY_DIP],
    [HandJoints.PINKY_DIP, HandJoints.PINKY_TIP]
]

bone_list = [
        [
            Joint.SPINE_CHEST, 
            Joint.SPINE_NAVEL
        ],
        [
            Joint.SPINE_NAVEL,
            Joint.PELVIS
        ],
        [
            Joint.SPINE_CHEST,
            Joint.NECK
        ],
        [
            Joint.NECK,
            Joint.HEAD
        ],
        [
            Joint.HEAD,
            Joint.NOSE
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_LEFT
        ],
        [
            Joint.CLAVICLE_LEFT,
            Joint.SHOULDER_LEFT
        ],
        [
            Joint.SHOULDER_LEFT,
            Joint.ELBOW_LEFT
        ],
        [
            Joint.ELBOW_LEFT,
            Joint.WRIST_LEFT
        ],
        [
            Joint.NOSE,
            Joint.EYE_LEFT
        ],
        [
            Joint.EYE_LEFT,
            Joint.EAR_LEFT
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_RIGHT
        ],
        [
            Joint.CLAVICLE_RIGHT,
            Joint.SHOULDER_RIGHT
        ],
        [
            Joint.SHOULDER_RIGHT,
            Joint.ELBOW_RIGHT
        ],
        [
            Joint.ELBOW_RIGHT,
            Joint.WRIST_RIGHT
        ],
        [
            Joint.NOSE,
            Joint.EYE_RIGHT
        ],
        [
            Joint.EYE_RIGHT,
            Joint.EAR_RIGHT
        ],
        [
            Joint.PELVIS,
            Joint.HIP_RIGHT
        ],
        [
            Joint.PELVIS,
            Joint.HIP_LEFT
        ],
        [
            Joint.HIP_RIGHT,
            Joint.KNEE_RIGHT
        ],
        [
            Joint.ANKLE_RIGHT,
            Joint.KNEE_RIGHT
        ],
        [
            Joint.ANKLE_RIGHT,
            Joint.FOOT_RIGHT
        ],
        [
            Joint.HIP_LEFT,
            Joint.KNEE_LEFT
        ],
        [
            Joint.ANKLE_LEFT,
            Joint.KNEE_LEFT
        ],
        [
            Joint.ANKLE_LEFT,
            Joint.FOOT_LEFT
        ]
]

#BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139,34,104)]
dotColors = [(0, 0, 139), (20,128,48), (71,130,170), (205,95,58), (205,150,205)]

frameCount = 0
shift = 7

@final
class ParadigmFrame(BaseFeature[ColorImageInterface]):
    """
    The final feature return the frame with gaze cones

    Input frames are `ColorImageInterface`, `GazeConesInterface` and `CameraCalibrationInterface`

    Output frame is `ColorImageInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        gestureLandmarks: BaseFeature[LandmarkInterface],
        bodyTracking: BaseFeature[BodyTrackingInterface], 
        calibration: BaseFeature[CameraCalibrationInterface],
        landmarks: bool = True  
    ):
        super().__init__(
            color, gestureLandmarks, bodyTracking, calibration
        )
        self.landmarks = landmarks

    def get_output(
        self,
        color: ColorImageInterface,
        gestureLandmarks: GestureConesInterface,
        bt: BodyTrackingInterface,
        calibration: CameraCalibrationInterface,
    ):
        if not color.is_new() or not gestureLandmarks.is_new() or not bt.is_new() or not calibration.is_new():
            return None

        # copy the frame, do not draw on the original frame
        output_frame = np.copy(color.frame)

        # adding body tracking and gesture landmarks
        if self.landmarks:
            for land in gestureLandmarks.landmarks:
                if land.joints is not None:
                    dotColor = dotColors[land.azureBodyId % len(dotColors)]; 
                    _color = (255,255,255) if land.handedness.value == "Left" else colors[land.azureBodyId % len(colors)]; 
                    for hand in hand_list:
                        cv.line(output_frame, land.joints[int(hand[0])], land.joints[int(hand[1])], color=_color, thickness=2)

                    for joint in land.joints:
                        cv.circle(output_frame, (joint[0], joint[1]), radius=2, thickness=2, color=dotColor)

            bt = fix_body_id(bt)
            for bodyIndex, body in enumerate(bt.bodies):  
                bodyId = int(body["wtd_body_id"])
                dotColor = dotColors[bodyId % len(dotColors)]; 
                _color = colors[bodyId % len(colors)]; 
                dictionary = {}
            
                for jointIndex, joint in enumerate(body["joint_positions"]):
                    bodyLocation = getPointSubcategory(Joint(jointIndex))
                    if(bodyLocation != BodyCategory.RIGHT_HAND and bodyLocation != BodyCategory.LEFT_HAND):
                        points2D, _ = cv.projectPoints(
                            np.array(joint), 
                            calibration.rotation,
                            calibration.translation,
                            calibration.camera_matrix,
                            calibration.distortion)  
                        
                        point = (int(points2D[0][0][0] * 2**shift),int(points2D[0][0][1] * 2**shift))
                        dictionary[Joint(jointIndex)] = point
                        cv.circle(output_frame, point, radius=15, color=dotColor, thickness=15, shift=shift)

                for bone in bone_list:
                    if(getPointSubcategory(bone[0]) == BodyCategory.RIGHT_ARM or getPointSubcategory(bone[1]) == BodyCategory.RIGHT_ARM):
                        cv.line(output_frame, dictionary[bone[0]], dictionary[bone[1]], color=(255,255,255), thickness=3, shift=shift)
                    else:
                        cv.line(output_frame, dictionary[bone[0]], dictionary[bone[1]], color=_color, thickness=3, shift=shift)

        # the default image shape is 1080 * 1920 * 3
        # reshape the frame
        output_frame = cv.resize(output_frame, (1280, 720))

        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)