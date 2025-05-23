from pathlib import Path
from mmdemo.features.outputs.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.outputs.paradigm_logging_feature import ParadigmLog

from mmdemo.demo import Demo
from mmdemo.features import (
    DisplayFrame,
    SaveVideo,
    DisplayFrame,
    SaveVideo,
    RealSenseCameraDevice
)
from mmdemo.features.realSense.features import create_real_sense_features

if __name__ == "__main__":
    # LIVE camera settings#####################################################
    color, depth, calibration = create_real_sense_features() #body_tracking, calibration
    ############################################################################

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(color),
            # SaveVideo(depth_output_frame_save, frame_rate=10, video_type="depth", delete_output=False),
            # SaveVideo(color_output_frame, frame_rate=10, video_type="color", delete_output=False),
            # ParadigmLog(gesture, body_tracking, depth, calibration, csv=True, fileName="paradigm"),
            #Log(transcriptions, stdout=True),
        ]
    )
    #demo.show_dependency_graph()
    demo.run()

