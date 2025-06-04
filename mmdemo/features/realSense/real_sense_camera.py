from dataclasses import dataclass
from typing import final
from mmdemo.base_feature import BaseFeature
import pyrealsense2 as rs
import numpy as np
import cv2

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import ColorImageInterface, DepthImageInterface

@dataclass
class _RealSenseInterface(BaseInterface):
    """
    Store output of Real Sense wrapper Device class. This is a private
    interface which is just used to pass information to the color, depth,
    and body tracking features.

    color -- color image in bgra
    depth -- depth image
    body_tracking -- body tracking output dict
    frame_count -- current frame
    camera_matrix -- camera matrix of camera
    distortion -- distortion of camera
    rotation -- rotation of camera
    translation -- translation of camera
    """

    color: np.ndarray
    depth: np.ndarray
    body_tracking: dict
    frame_count: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

@final
class RealSenseCameraDevice(BaseFeature):
    """
    Detect the real sense camera data.

    Input interfaces are none

    Output interface is `ColorImageInterface`, `DepthImageInterface`, `BodyTrackingInterface`, `CameraCalibrationInterface`

    """

    def __init__(
        self
    ):
        super().__init__()

    def initialize(self):
        self.frameCount = -1

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # camera settings
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # video playback
        # rs.config.enable_device_from_file(config, "D:\Weights_Task\Data\Fib_weights_original_videos\Group_01-master.mkv") # Replace with your file path

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        self.profile = self.pipeline.start(config)

    def get_output(
        self
    ):
        self.frameCount+=1

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        pose = frames.first_or_default(rs.stream.pose)
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        # Get extrinsics from depth to color
        depth_to_color_extrinsic = depth_profile.get_extrinsics_to(color_profile)
        intrinsics = color_profile.get_intrinsics()
        matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy],
                  [0, 0, 1]])
        distortionCeoffs = np.array(intrinsics.coeffs)

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(1)

        return _RealSenseInterface(
            color=color_image,
            depth=depth_image,
            body_tracking={},
            frame_count=self.frameCount,
            camera_matrix=matrix,
            distortion=distortionCeoffs,
            rotation=depth_to_color_extrinsic.rotation,
            translation=depth_to_color_extrinsic.translation)