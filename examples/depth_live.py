from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    DisplayFrame,
    DepthFrame
)

if __name__ == "__main__":
    # azure kinect features from camera
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA, camera_index=0
    )

    # TODO add custom gesture and pose that return the landmarks to draw on the depth output

    output_frame = DepthFrame(depth)

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

