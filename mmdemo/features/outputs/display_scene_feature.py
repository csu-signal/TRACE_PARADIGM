from threading import Lock
from queue import SimpleQueue
import random
import time
from typing import final

import cv2 as cv

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, EmptyInterface, SceneInterface
import pyrender
import numpy as np


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
        self.mn, self.cn, self.ln = None, None, None

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

        self.viewer.render_lock.acquire()
        # self.scene.clear()

        zoom_factor = 1.0
        angle_x = 0.0
        angle_y = 0.0
        rotate_step = np.radians(15)
        zoom_step = 0.1

        mesh_extent = np.max(mesh.mesh_scene.bounding_box.extents)
        base_distance = mesh_extent * 2.5
        camera_distance = base_distance * zoom_factor

        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x), 0],
            [0, np.sin(angle_x), np.cos(angle_x), 0],
            [0, 0, 0, 1]
        ])
        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
            [0, 0, 0, 1]
        ])
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, camera_distance],
            [0, 0, 0, 1]
        ])
        cam_pose = R_y @ R_x @ T

        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh.mesh_scene)
        camera_obj = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        # scene.add(camera_obj, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        # scene.add(light, pose=cam_pose)
        # scene = pyrender.Scene()

        if self.mn is not None:
            self.scene.remove_node(self.mn)
            self.scene.remove_node(self.cn)
            self.scene.remove_node(self.ln)

        self.mn = pyrender.Node(mesh=mesh_pyrender)
        self.cn = pyrender.Node(camera=camera_obj)
        self.ln = pyrender.Node(light=light)
        self.scene.add_node(self.mn)
        self.scene.add_node(self.cn)
        self.scene.add_node(self.ln)

        self.viewer.render_lock.release()


        return EmptyInterface()

    def is_done(self):
        return (
            self.window_should_be_up and not self.viewer.is_active
        )
