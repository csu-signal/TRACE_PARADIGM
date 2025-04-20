import random
from typing import final

import cv2 as cv

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, EmptyInterface, SceneInterface
import pyrender


@final
class DisplayScene(BaseFeature[EmptyInterface]):
    """
    Show a pyrender.Scene frame with pyrender.Viewer. The demo will exit once
    the window is closed.

    Input interface is `SceneInterface`

    Output interface is `EmptyInterface`
    """

    def __init__(
        self,
        scene: BaseFeature[SceneInterface],
    ):
        super().__init__(scene)

    def initialize(self):
        self.window_should_be_up = False

    def get_output(
        self,
        scene: SceneInterface,
    ):
        if not scene.is_new():
            self.window_should_be_up = False
            return None
       
        # Open pyrender window
        self.viewer = pyrender.Viewer(scene.mesh_scene, use_raymond_lighting=True, run_in_thread=False) #TODO figure out threading, does run in thread = True cause a memory leak?
        self.window_should_be_up = True

        return EmptyInterface()

    def is_done(self):
        return (
            self.window_should_be_up and not self.viewer.is_active
        )
