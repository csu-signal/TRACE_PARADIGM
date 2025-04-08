import warnings
from pathlib import Path
from typing import final

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
)
from mmdemo.interfaces.data import Cone, Handedness, Landmarks
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d


@final
class GestureLandmarks(BaseFeature[LandmarkInterface]):
    """
    Detect when and where hand landmarks are located.

    Input interfaces are `ColorImageInterface`, `DepthImageInterface`,
    `BodyTrackingInterface`, `CameraCalibrationInterface`

    Output interface is `LandmarkInterface`
    """

    HAND_BOUNDING_BOX_WIDTH = 192
    HAND_BOUNDING_BOX_HEIGHT = 192

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        bt: BaseFeature[BodyTrackingInterface],
        calibration: BaseFeature[CameraCalibrationInterface]
    ):
        super().__init__(color, depth, bt, calibration)

    def initialize(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            static_image_mode=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0,
        )

    def get_output(
        self,
        color: ColorImageInterface,
        depth: DepthImageInterface,
        bt: BodyTrackingInterface,
        calibration: CameraCalibrationInterface,
    ):
        if not color.is_new() or not depth.is_new() or not bt.is_new():
            return None

        landmark_output = []
        bt = fix_body_id(bt)
        for _, body in enumerate(bt.bodies):
            for handedness in (Handedness.Left, Handedness.Right):
                # loop through both hands of all bodies

                # create a box around the hand using azure kinect info
                avg = get_average_hand_pixel(body, calibration, handedness)
                offset = (
                    np.array(
                        [self.HAND_BOUNDING_BOX_WIDTH, self.HAND_BOUNDING_BOX_HEIGHT]
                    )
                    / 2
                )
                box = np.array([avg - offset, avg + offset], dtype=np.int64)
                if (
                    box[0][0] < 0
                    or box[0][1] < 0
                    or box[1][0] >= color.frame.shape[1]
                    or box[1][1] >= color.frame.shape[0]
                ):
                    return None

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    joints = self.find_landmarks(color, box)
                    landmark_output.append(
                        Landmarks(
                            joints=joints,
                            azureBodyId=body["wtd_body_id"],
                            handedness=handedness
                        )
                    )

        return LandmarkInterface(
            landmarks=landmark_output
        )

    def find_landmarks(
        self,
        color: ColorImageInterface,
        box: np.ndarray,
    ) -> np.ndarray | None:
        """
        Find the hand landmarks inside of a given box of the frame.

        Arguments:
        color -- color image interface
        box -- [upper left, lower right] where both points are (x,y)

        Returns:
        the hand landmark array
        """
        # subframe containing only the hand
        hand_frame = color.frame[box[0][1] : box[1][1], box[0][0] : box[1][0]]
        h, w, c = hand_frame.shape

        # run mediapipe
        mediapipe_results = self.hands.process(hand_frame)

        # if we don't have results for hand landmarks, then exit
        if not mediapipe_results.multi_hand_landmarks:
            return None

        landmarks = []
        for index, handslms in enumerate(mediapipe_results.multi_hand_landmarks):
            for lm in handslms.landmark:
                lmx = int(lm.x * w) + box[0][0]
                lmy = int(lm.y * h) + box[0][1]
                landmarks.append([lmx, lmy])
        return landmarks

            