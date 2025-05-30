from pathlib import Path
from datetime import datetime

from mmdemo.features.gesture.gesture_landmark_feature import GestureLandmarks
from mmdemo.features.outputs.depth_frame_feature import DepthFrame
from mmdemo.features.outputs.display_scene_feature import DisplayScene
from mmdemo.features.outputs.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.outputs.paradigm_frame_feature import ParadigmFrame
from mmdemo.features.outputs.paradigm_logging_feature import ParadigmLog

from mmdemo.demo import Demo
from mmdemo.features import (
    DisplayFrame,
    SaveVideo,
    DisplayFrame,
    SaveVideo,
    RealSenseCameraDevice
)
from mmdemo.features.pose.cliff_pose_feature import CliffPose
from mmdemo.features.realSense.features import create_real_sense_features

if __name__ == "__main__":
    save_dir_prefix = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
    # LIVE camera settings#####################################################
    color, depth, calibration, body_tracking = create_real_sense_features() #body_tracking, calibration
    ############################################################################
    gesture = GestureLandmarks(color, depth, body_tracking, calibration)

    displayScene = CliffPose(color, depth, body_tracking, calibration)

    color_output_frame = ParadigmFrame(
        color=color,
        gestureLandmarks=gesture,
        bodyTracking = body_tracking,
        calibration=calibration,
        landmarks = False
    )
    depth_output_frame = DepthFrame(depth, gesture, body_tracking, calibration, landmarks=False)
    # run demo and show output
    demo = Demo(
        targets=[
            # DisplayFrame(color),
            DisplayFrame(color_output_frame),
            DisplayFrame(depth_output_frame),
            DisplayScene(displayScene, record=False, save_dir_prefix=save_dir_prefix),
            # SaveVideo(depth_output_frame_save, frame_rate=10, video_type="depth", delete_output=False),
            # SaveVideo(color_output_frame, frame_rate=10, video_type="color", delete_output=False, save_dir_prefix=save_dir_prefix),
            # ParadigmLog(gesture, body_tracking, depth, calibration, csv=True, fileName="paradigm"),
            #Log(transcriptions, stdout=True),
        ]
    )
    #demo.show_dependency_graph()
    demo.run()

