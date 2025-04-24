from threading import Lock
from queue import SimpleQueue
import random
import time
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
        self.scene = pyrender.Scene()
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
        # self.viewer.start()
        # self.viewer.

    def get_output(
        self,
        mesh: SceneInterface,
    ):
        if not mesh.is_new():
            self.window_should_be_up = False
            return None
       
        # Open pyrender window
        # self.viewer = pyrender.Viewer(scene.mesh_scene, use_raymond_lighting=True, run_in_thread=False) #TODO figure out threading, does run in thread = True cause a memory leak?
        self.window_should_be_up = True

        #? Do we need the while loop? It seems like the viewer is already running in a thread.
        # while self.viewer.is_active:
        #     # Acquire the render lock before updating the scene
        #     self.viewer.render_lock.acquire()
        #     try:
        #         self.sceneLock.acquire()
        #         if not self.sceneQueue.empty():
        #             # Update your scene (e.g., move the mesh, change the camera)
        #             # self.scene.clear()
        #             mesh_scene = self.sceneQueue.get()
        #             self.scene.add(mesh_scene)
        #     finally:
        #     # Release the render lock after updating
        #         self.sceneLock.release() 
        #         self.viewer.render_lock.release()
        start_time = time.time()
        self.viewer.render_lock.acquire()
        # self.scene.clear()
        self.scene.add(mesh.mesh_scene, name="patient_mesh")
        self.viewer.render_lock.release()
        end_time = time.time()
        print("Time to render scene= ", (end_time-start_time))


        return EmptyInterface()

    def is_done(self):
        return (
            self.window_should_be_up and not self.viewer.is_active
        )
