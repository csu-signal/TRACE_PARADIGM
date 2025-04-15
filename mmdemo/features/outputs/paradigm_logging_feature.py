import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import final
import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.features.gesture.helpers import fix_body_id
from mmdemo.interfaces import BodyTrackingInterface, CameraCalibrationInterface, DepthImageInterface, EmptyInterface, GazeConesInterface, GestureConesInterface, HciiGestureConesInterface, HciiSelectedObjectsInterface, LandmarkInterface
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.interfaces.data import HciiSelectedObjectInfo
from mmdemo.utils.coordinates import pixel_to_camera_3d, CoordinateConversionError

@final
class ParadigmLog(BaseFeature[EmptyInterface]):
    """
    Log output to stdout and/or csv files. If logging to csv files,
    headers called "log_frame" and "log_time" will be added to represent
    when the output is happening.

    Input interfaces can be any number of `BaseInterfaces`

    Output interface is `EmptyInterface`

    Keyword arguments:
        stdout -- if the interfaces should be printed
        gesture -- the paradigm gesture feature
        bodyTracking -- the paradigm body tracking feature
        csv -- if the interfaces should be saved to csvs
        files -- a list of file names inside of the output directory,
                this should be in the same order as input features
        output_dir -- output directory if logging to files
    """

    def __init__(
        self, gesture, bodyTracking, depth, calibration, stdout=False, csv=False, fileName=None, output_dir=None
    ) -> None:
        self.stdout = stdout
        self.csv = csv
        self.fileName = f"{fileName}.csv"
        self._out_dir = output_dir

        super().__init__(gesture, bodyTracking, depth, calibration)

    def initialize(self):
        if self.csv:
            # create output directory
            if self._out_dir is not None:
                self.output_dir = Path(self._out_dir)
            else:
                self.output_dir = Path(
                    "logging-output-"
                    + datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
                )
            os.makedirs(self.output_dir, exist_ok=True)

            # create output files
            file = self.output_dir / self.fileName
            assert (
                not file.is_file()
            ), f"A logging file already exists and cannot be overwritten ({file})"
            file.touch()
            self.needs_header = True

        self.frame = 0

    def get_output(self, *args):
        logged_something = False
        if(len(args) == 4):
            gestureLandmarks = args[0]
            bt = args[1]
            depth = args[2]
            calibration = args[3]
            if gestureLandmarks.is_new() or bt.is_new() or depth.is_new() or calibration.is_new():
                self.log(gestureLandmarks, bt, depth, calibration)
                logged_something = True

        self.frame += 1
        return EmptyInterface() if logged_something else None

    def log(self, gestureLandmarks: LandmarkInterface, bt: BodyTrackingInterface, depth: DepthImageInterface, calibration: CameraCalibrationInterface):
        if self.stdout:
            print(f"(frame {self.frame:05})", gestureLandmarks, bt)

        if self.csv:
            file: Path = self.output_dir / self.fileName
            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                header_row = ["frame_index"]
                output_row = [self.frame]

                header_row.append('aspect_ratio')
                h, w = depth.frame.shape
                output_row.append(f"{w}x{h}")

                header_row.append("patient_body_joints")
                header_row.append("practitioner_body_joints")

                practitioner = []
                patient = []
                bt = fix_body_id(bt)
                for bodyIndex, body in enumerate(bt.bodies):  
                    bodyId = int(body["wtd_body_id"])
                    for jointIndex, joint in enumerate(body["joint_positions"]):
                        points2D, _ = cv.projectPoints(
                            np.array(joint), 
                            calibration.rotation,
                            calibration.translation,
                            calibration.camera_matrix,
                            calibration.distortion) 
                        point = (int(points2D[0][0][0]),int(points2D[0][0][1]))  
                        if(bodyId == 1):
                            patient.append(point)
                        if(bodyId == 2):
                            practitioner.append(point)
                
                output_row.append(patient)
                output_row.append(practitioner)

                header_row.append("patient_right_joints")
                header_row.append("patient_left_joints")
                header_row.append("practitioner_right_joints")
                header_row.append("practitioner_left_joints")

                practitioner_r = []
                practitioner_l = []
                patient_r = []
                patient_l = []
                for land in gestureLandmarks.landmarks:
                    # if land.joints is not None:
                    #     joints3D = []
                    #     try:
                    #         for joint in land.joints:
                    #             joints3D.append(pixel_to_camera_3d(joint, depth, calibration))
                    if(land.azureBodyId == 1):
                        if(land.handedness.value == "Right"):
                            patient_r = land.joints
                        if(land.handedness.value == "Left"):
                            patient_l = land.joints

                    if(land.azureBodyId == 2):
                        if(land.handedness.value == "Right"):
                            practitioner_r = land.joints
                        if(land.handedness.value == "Left"):
                            practitioner_l = land.joints

                        # except CoordinateConversionError:
                        #     pass

                output_row.append(patient_r)
                output_row.append(patient_l)
                output_row.append(practitioner_r)
                output_row.append(practitioner_l)

                header_row.append('rotation')
                header_row.append('translation')
                header_row.append('camera_matrix')
                header_row.append('distortion')

                output_row.append(calibration.rotation)
                output_row.append(calibration.translation)
                output_row.append(calibration.camera_matrix)
                output_row.append(calibration.distortion)

                if self.needs_header:
                    writer.writerow(header_row)
                    self.needs_header = False
                writer.writerow(output_row)
