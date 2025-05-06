from datetime import datetime
from pathlib import Path
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
import trimesh
import trimesh.transformations as tf

JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]


STATE_CONFIG = {
    1: {
        'text': "Check knees visually for redness",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP LKnee")], joints[JOINT_NAMES.index("OP RKnee")]]  # Left and Right knees
    },
    2: {
        'text': "Check for redness and swelling",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LKnee")] + joints[JOINT_NAMES.index("OP LAnkle")]) / 2.0,  # Left calf (midpoint between knee and ankle)
            (joints[JOINT_NAMES.index("OP RKnee")] + joints[JOINT_NAMES.index("OP RAnkle")]) / 2.0   # Right calf (midpoint between knee and ankle)
        ]
    },
    3: {
        'text': "Check between toes",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LSmallToe")] + joints[JOINT_NAMES.index("OP LBigToe")]) / 2.0,  # Left toe (midpoint between big and small toe)
            (joints[JOINT_NAMES.index("OP RSmallToe")] + joints[JOINT_NAMES.index("OP RBigToe")]) / 2.0   # Right toe (midpoint between big and small toe)
        ]
    },
    4: {
        'text': "Check for swelling",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP RHeel")], joints[JOINT_NAMES.index("OP LHeel")]]  # Left and Right heels
    }
}

def create_arrow(start, end, shaft_radius=0.005, head_radius=0.01, head_length=0.02, sections=20):
    """
    Create an arrow mesh from a start point (tail) to an end point (head) using trimesh.
    The arrow is built from a cylinder (shaft) and a cone (head).
    """
    vec = end - start
    total_length = np.linalg.norm(vec)
    if total_length < 1e-6:
        return None
    direction = vec / total_length

    # Reserve space for the arrow head.
    shaft_length = max(total_length - head_length, total_length * 0.8)
    head_length = total_length - shaft_length

    # Create the shaft as a cylinder along the Z-axis.
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length, sections=sections)
    shaft.apply_translation([0, 0, shaft_length / 2.0])

    # Create the head as a cone along the Z-axis.
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=sections)
    head.apply_translation([0, 0, shaft_length + head_length / 2.0])

    # Combine shaft and head.
    arrow = trimesh.util.concatenate([shaft, head])

    # Align the arrow (default along Z) with the desired direction.
    z_axis = np.array([0, 0, 1])
    rot_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    if rot_matrix is None:
        rot_matrix = np.eye(3)
    elif rot_matrix.shape == (4, 4):
        rot_matrix = rot_matrix[:3, :3]
    
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_matrix
    arrow.apply_transform(T_rot)

    # Translate so that its base (tail) is at the start position.
    arrow.apply_translation(start)
    return arrow




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
        record: bool = False,
        save_dir_prefix: str = None,
    ):
        super().__init__(scene)
        self.record = record
        self.save_dir_prefix = save_dir_prefix


    def _increase_x_rotation(self):
        self.angle_x += self.rotate_step
    def _decrease_x_rotation(self):
        self.angle_x -= self.rotate_step
    def _increase_y_rotation(self):
        self.angle_y += self.rotate_step
    def _decrease_y_rotation(self):
        self.angle_y -= self.rotate_step

    def _increase_zoom(self):
        self.zoom_factor *= (1 - self.zoom_step)
    def _decrease_zoom(self):
        self.zoom_factor *= (1 + self.zoom_step)
    
    def _state_change(self, key):
        self.current_state = int(chr(key))

    def initialize(self):
        self.window_should_be_up = False
        '''
        self.mn => patient's body mesh node
        self.an => arrow mesh node pointing to some body part
        self.cn => camera mesh node
        self.ln => lighting mesh node
        self.pn => US probe centroid's node
        '''
        self.mn, self.an, self.cn, self.ln, self.pn = None, None, None, None, None

        self.current_state = 1
        self.zoom_factor = 1.0
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.rotate_step = np.radians(15)
        self.zoom_step = 0.1

        # _registered_keys = {
        #     "w": lambda: self._increase_x_rotation(),
        #     "s": (self._decrease_x_rotation, [self]),
        #     "a": (self._decrease_y_rotation, [self]),
        #     "d": (self._increase_y_rotation, [self]),
        #     "=": (self._increase_zoom, [self]),
        #     "-": (self._decrease_zoom, [self]),
        #     "1": (self._state_change, [self, "1"]),
        #     "2": (self._state_change, [self], "2"),
        #     "3": (self._state_change, [self], "3"),
        #     "4": (self._state_change, [self], "4")
        # }

        # axis_trimesh = trimesh.creation.axis(
        #     origin_size = 0.03,     # little cube at the origin
        #     axis_length = 0.3)      # length of each arrow (metres)

        # axis_mesh = pyrender.Mesh.from_trimesh(axis_trimesh, smooth=False)
        # axis_node = pyrender.Node(mesh=axis_mesh)

        self.scene = pyrender.Scene()
        # self.scene.add_node(axis_node)
        # self.scene.set_pose(axis_node, np.eye(4))
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True, viewer_flags={"record": self.record},)


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

        # key = cv.waitKey(1)
        # if key != -1:
        #     if key == ord('w'):
        #         self.angle_x -= self.rotate_step
        #     elif key == ord('s'):
        #         self.angle_x += self.rotate_step
        #     elif key == ord('a'):
        #         self.angle_y -= self.rotate_step
        #     elif key == ord('d'):
        #         self.angle_y += self.rotate_step
        #     elif key in [ord('+'), ord('=')]:
        #         self.zoom_factor *= (1 - self.zoom_step)
        #     elif key in [ord('-'), ord('_')]:
        #         self.zoom_factor *= (1 + self.zoom_step)
        #     elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        #         self.current_state = int(chr(key))
        #         print(f"State changed to {self.current_state}")


        mesh_extent = np.max(mesh.mesh_scene.bounding_box.extents)
        base_distance = mesh_extent * 2.5
        camera_distance = base_distance * self.zoom_factor

        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(self.angle_x), -np.sin(self.angle_x), 0],
            [0, np.sin(self.angle_x), np.cos(self.angle_x), 0],
            [0, 0, 0, 1]
        ])
        R_y = np.array([
            [np.cos(self.angle_y), 0, np.sin(self.angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(self.angle_y), 0, np.cos(self.angle_y), 0],
            [0, 0, 0, 1]
        ])
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, camera_distance],
            [0, 0, 0, 1]
        ])
        cam_pose = R_y @ R_x @ T
        R_flip = tf.rotation_matrix(2*np.pi, [1, 1, 0])

        ### setting joints to be displayed
        mesh_centroid = mesh.mesh_scene.bounding_box.centroid.copy()
        joints_centered = mesh.smpl_joints - mesh_centroid
        if self.current_state in STATE_CONFIG:
            targets = STATE_CONFIG[self.current_state]['targets'](joints_centered)
            for target in targets:
                # Define a constant tail offset. Adjust this as necessary.
                offset = np.array([0.0, -0.02, -0.4])
                head_offset = np.array([0.0, 0.0, -0.1])
                target = target +head_offset
                tail = target + offset

                arrow_mesh = create_arrow(tail, target,  shaft_radius=0.01,head_radius=0.03, head_length=0.08)
                if arrow_mesh is not None:
                    arrow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.0, 0.0, 0.0, 1.0))
                    arrow_mesh.apply_transform(R_flip)
                    arrow_pyrender = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_material, smooth=False)


        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh.mesh_scene)
        camera_obj = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        if self.mn is not None:
            self.scene.remove_node(self.mn)
            self.scene.remove_node(self.an)
            self.scene.remove_node(self.cn)
            self.scene.remove_node(self.ln)
        if self.scene.has_node(self.pn):
            self.scene.remove_node(self.pn)

        self.mn = pyrender.Node(mesh=mesh_pyrender)
        self.cn = pyrender.Node(camera=camera_obj)
        self.ln = pyrender.Node(light=light)
        self.an = pyrender.Node(mesh=arrow_pyrender)
        self.scene.add_node(self.mn)
        self.scene.add_node(self.an)
        self.scene.add_node(self.cn)
        self.scene.set_pose(self.cn, cam_pose)
        self.scene.add_node(self.ln)
        self.scene.set_pose(self.ln, cam_pose)

        if mesh.probe_centroid is not None:
            print("probe is detected", mesh.probe_centroid)
            sphere_trimesh = trimesh.creation.icosphere(subdivisions=3, radius=10)
            sphere_trimesh.visual.vertex_colors = [255, 0, 0, 255]   # RGBA red
            sphere = pyrender.Mesh.from_trimesh(sphere_trimesh, smooth=False)
            self.pn = pyrender.Node(sphere)
            self.scene.add_node(self.pn)
            self.scene.set_pose(self.pn, np.eye(4))
            self.scene.set_pose(self.pn, np.block([
                [np.eye(3), mesh.probe_centroid.reshape(3,1)],
                [np.zeros((1,3)), 1]
            ]))
            # self.pn = self.scene.add(sphere, pose=np.eye(4))          # identity pose
            # self.scene.set_pose(self.pn, )
            # print()
        # else:
        #     print("probe is not detected")
        #     self.pn = None

        self.viewer.render_lock.release()


        return EmptyInterface()

    def is_done(self):
        return (
            self.window_should_be_up and not self.viewer.is_active
        )

    def finalize(self):
       
       self.viewer.close_external()
       if self.record:
        video_name = Path(
                f"output/{self.save_dir_prefix}/smpl_mesh" + ".gif"
                )
        self.viewer.save_gif(video_name)
