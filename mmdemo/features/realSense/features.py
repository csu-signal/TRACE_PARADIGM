"""
Features which can be used as dependencies in a demo
"""

from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.features import RealSenseCameraDevice
from mmdemo.features.realSense.real_sense_camera import _RealSenseInterface
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface
)


@final
class RealSenseColor(BaseFeature):
    """
    Feature to get color images from Real Sense.

    The input interface is `_RealSenseInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `ColorImageInterface`.
    """

    def get_output(
        self, real_input: _RealSenseInterface
    ) -> ColorImageInterface | None:
        if not real_input.is_new():
            return None
        return ColorImageInterface(
            frame_count=real_input.frame_count, frame=real_input.color
        )


@final
class RealSenseDepth(BaseFeature):
    """
    Feature to get depth images from Real Sense.

    The input interface is `_RealSenseInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `DepthImageInterface`.
    """

    def get_output(
        self, real_input: _RealSenseInterface
    ) -> DepthImageInterface | None:
        if not real_input.is_new():
            return None
        return DepthImageInterface(
            frame_count=real_input.frame_count, frame=real_input.depth
        )


# @final
# class AzureKinectBodyTracking(BaseFeature):
#     """
#     Feature to get body tracking info from Real Sense.

#     The input interface is `_RealSenseInterface`, which is a private
#     interface that is created using helper functions.

#     The output interface is `BodyTrackingInterface`.
#     """

#     def get_output(
#         self, real_input: _RealSenseInterface
#     ) -> BodyTrackingInterface | None:
#         if not real_input.is_new():
#             return None
#         return BodyTrackingInterface(
#             bodies=real_input.body_tracking["bodies"],
#             timestamp_usec=real_input.body_tracking["timestamp_usec"],
#         )


@final
class RealSenseCameraCalibration(BaseFeature):
    """
    Feature to get camera calibration info from Real Sense.

    The input interface is `_RealSenseInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `CameraCalibrationInterface`.
    """

    def get_output(
        self, real_input: _RealSenseInterface
    ) -> CameraCalibrationInterface | None:
        if not real_input.is_new():
            return None
        return CameraCalibrationInterface(
            camera_matrix=real_input.camera_matrix,
            distortion=real_input.distortion,
            rotation=real_input.rotation,
            translation=real_input.translation,
        )


def create_real_sense_features(
    # device_type: DeviceType,
    # *,
    # camera_index: int | None = None,
    # mkv_path: str | Path | None = None,
    # mkv_frame_rate: int | None = 30,
    # playback_frame_rate: int | None = 5,
    # playback_end_seconds: int | None = None
):
    # """ TODO update documentation
    # Returns 4 features which output `ColorImageInterface`, `DepthImageInterface`,
    # `BodyTrackingInterface`, and `CameraCalibrationInterface` using information
    # from an Real Sense camera or playback mkv file.

    # Arguments:
    # `device_type` -- an instance of the DeviceType enum specifying if the
    # data should come from a camera or playback

    # Keyword Arguments:
    # `camera_index` -- the index of the Real Sense camera, used for `DeviceType.CAMERA`
    # `mkv_path` -- the path to an Real Sense playback mkv file, used for `DeviceType.PLAYBACK`
    # `mkv_frame_rate` -- frame rate of the mkv file, default 30, used for `DeviceType.PLAYBACK`
    # `playback_frame_rate` -- simulated frame rate of the playback, default 5, this can reduce the number of frames which need to be processed, used for `DeviceType.PLAYBACK`. This does not change the frame counts of the features, it just ignores a certain number of frames.
    # `playback_end_seconds` -- the number of seconds to end playback after or None, default None, used for `DeviceType.PLAYBACK`
    # """
    input_feature = RealSenseCameraDevice()

    color = RealSenseColor(input_feature)
    depth = RealSenseDepth(input_feature)
    # body_tracking = AzureKinectBodyTracking(input_feature)
    calibration = RealSenseCameraCalibration(input_feature)

    return color, depth, calibration
