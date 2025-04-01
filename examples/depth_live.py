from pathlib import Path
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    DisplayFrame,
    DepthFrame,
    GestureLandmarks
)

# mkv path for WTD group
WTD_MKV_PATH = (
    "D:/Weights_Task/Data/Fib_weights_original_videos/Group_{0:02}-master.mkv"
)

# Number of frames to evaluate per second. This must
# be a divisor of 30 (the true frame rate). Higher rates
# will take longer to process.
PLAYBACK_FRAME_RATE = 5

# The number of seconds of the recording to process
WTD_END_TIMES = {
    1: 5 * 60 + 30,
    2: 5 * 60 + 48,
    3: 8 * 60 + 3,
    4: 3 * 60 + 31,
    5: 4 * 60 + 34,
    6: 5 * 60 + 3,
    7: 8 * 60 + 30,
    8: 6 * 60 + 28,
    9: 3 * 60 + 46,
    10: 6 * 60 + 51,
    11: 2 * 60 + 19,
}

if __name__ == "__main__":
    # LIVE camera settings#####################################################
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA, camera_index=0
    )
    ############################################################################

    # POST process entry settings (for development/debugging)##################
    # group = 1

    # # load azure kinect features from file
    # color, depth, body_tracking, calibration = create_azure_kinect_features(
    #     DeviceType.PLAYBACK,
    #     mkv_path=Path(WTD_MKV_PATH.format(group)),
    #     playback_end_seconds=WTD_END_TIMES[group],
    #     playback_frame_rate=PLAYBACK_FRAME_RATE,
    # )
    ############################################################################

    gesture = GestureLandmarks(color, depth, body_tracking, calibration)

    output_frame = DepthFrame(depth, gesture, body_tracking, calibration)

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            #SaveVideo(output_frame, frame_rate=10),
            #Log(friction, csv=True),
            #Log(transcriptions, stdout=True),
        ]
    )
    #demo.show_dependency_graph()
    demo.run()

