from datetime import datetime
import math
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
from pyrender.constants import TextAlign


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


def _increase_x_rotation(ds):
    ds.angle_x += ds.rotate_step
def _decrease_x_rotation(ds):
    ds.angle_x -= ds.rotate_step
def _increase_y_rotation(ds):
    ds.angle_y += ds.rotate_step
def _decrease_y_rotation(ds):
    ds.angle_y -= ds.rotate_step

def _increase_zoom(ds):
    ds.zoom_factor *= (1 - ds.zoom_step)
def _decrease_zoom(ds):
    ds.zoom_factor *= (1 + ds.zoom_step)

def _state_change(ds, key):
    ds.current_state = int(chr(key))



class ProximityTracker:
    def __init__(self, num_leg_keypoints, radius):
        self.radius_squared = radius ** 2
        self.num_leg_keypoints = num_leg_keypoints
        self.proximity_counts = np.zeros(num_leg_keypoints, dtype=int)

    def update(self, probe_position, leg_keypoints):
        """
        Update proximity counts using a new frame.
        
        Args:
            probe_position: (x, y, z) tuple or np.array
            leg_keypoints: list or np.array of shape (num_keypoints, 3)
        """

        print(f"leg_keypoints: {leg_keypoints.shape}")
        print(f"probe_position: {probe_position.shape}")
        deltas = np.array(leg_keypoints) - np.array(probe_position)
        distances_squared = np.sum(deltas ** 2, axis=1)
        close = distances_squared < self.radius_squared
        print(f"close: {close.shape}")

        self.proximity_counts += close.astype(int)

    def get_counts(self):
        return dict(enumerate(self.proximity_counts))


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
        self.proximity_tracker = ProximityTracker(6, 0.2)


    

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
        self.custom_markers = set()
        self.frame_count = 0

        self.scanning_progress = {"thigh": 0, "shin": 0} #JACK123

        _registered_keys = {
            "w": lambda _ : _increase_x_rotation(self),
            "s": lambda _ : _decrease_x_rotation(self),
            "a": lambda _ : _decrease_y_rotation(self),
            "d": lambda _ : _increase_y_rotation(self),
            "=": lambda _ : _increase_zoom(self),
            "-": lambda _ : _decrease_zoom(self),
            "1": lambda _ : _state_change(self, "1"),
            "2": lambda _ : _state_change(self, "2"),
            "3": lambda _ : _state_change(self, "3"),
            "4": lambda _ : _state_change(self, "4")
        }

        axis_trimesh = trimesh.creation.axis(
            origin_size = 0.03,     # little cube at the origin
            axis_length = 0.3)      # length of each arrow (metres)

        axis_mesh = pyrender.Mesh.from_trimesh(axis_trimesh, smooth=False)
        axis_node = pyrender.Node(mesh=axis_mesh)

        self.scene = pyrender.Scene()
        self.scene.clear()
        self.caption = caption = [dict(
            text     = STATE_CONFIG[self.current_state]['text'],
            location = TextAlign.TOP_CENTER,
            font_name = r"C:\Windows\Fonts\arial.ttf",
            font_pt   = 30,
            color    = (0.,1.,0.,1.),
            scale    = 1.0)]
        
        cam_pose = [
            [ 0.94023129, -0.27320049,  0.2032895,   0.69486399],
            [-0.17385694, -0.89841259, -0.40327233, -0.97562134],
            [ 0.29281204,  0.34382598, -0.89221343, -2.32217645],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ]
        camera_obj = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.cn = pyrender.Node(camera=camera_obj)
        self.ln = pyrender.Node(light=light)
        self.scene.add_node(self.cn)
        self.scene.set_pose(self.cn, cam_pose)
        self.scene.add_node(self.ln)
        self.scene.set_pose(self.ln, cam_pose)

        # self.scene.add_node(axis_node)
        # self.scene.set_pose(axis_node, np.eye(4))
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True, viewer_flags={"record": self.record, 'caption': self.caption,}, registered_keys=_registered_keys)


    def get_output(
        self,
        mesh: SceneInterface,
    ):
        if not mesh.is_new() or mesh.mesh_scene is None:
            self.window_should_be_up = False
            return None
       
        # Open pyrender window
        # self.viewer = pyrender.Viewer(scene.mesh_scene, use_raymond_lighting=True, run_in_thread=False) #TODO figure out threading, does run in thread = True cause a memory leak?
        self.window_should_be_up = True

        self.viewer.render_lock.acquire()

        self.mesh = mesh
        mesh_extent = np.max(self.mesh.mesh_scene.bounding_box.extents)
        base_distance = mesh_extent * 2.5
        # camera_distance = base_distance * self.zoom_factor

        R_flip = tf.rotation_matrix(2*np.pi, [1, 1, 0])

        ### setting joints to be displayed
        mesh_centroid = self.mesh.mesh_scene.bounding_box.centroid.copy()
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

        # mesh.mesh_scene.apply_transform(R_flip)
        # mesh_pyrender = pyrender.Mesh.from_trimesh(mesh.mesh_scene)
        camera_obj = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

        if self.scene.has_node(self.mn):
            self.scene.remove_node(self.mn)

        if self.scene.has_node(self.pn):
            self.scene.remove_node(self.pn)

        # self.mn = pyrender.Node(mesh=mesh_pyrender)
        # self.an = pyrender.Node(mesh=arrow_pyrender)
        # self.scene.add_node(self.mn)
        # self.scene.add_node(self.an)

        if mesh.probe_centroid is not None:
            print("probe is detected", mesh.probe_centroid)
            sphere_trimesh = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
            sphere_trimesh.visual.vertex_colors = [255, 0, 0, 255]   # RGBA red
            sphere = pyrender.Mesh.from_trimesh(sphere_trimesh, smooth=False)
            self.pn = self.scene.add(sphere, pose=np.eye(4))          # identity pose
            self.scene.set_pose(self.pn, np.block([
                [np.eye(3), mesh.probe_centroid.reshape(3,1)],
                [np.zeros((1,3)), 1]
            ]))

            # point_to_line_distance_3d = self.point_to_line_distance_3d(mesh.probe_centroid, mesh.smpl_joints[12], mesh.smpl_joints[13])

            # update the tracker
            if mesh.pelvis_coords is not None:
                self.add_custom_marker_az(mesh.smpl_joints, 12, 13, 0.2, -90, mesh.probe_centroid, add_shpere=False, sphere_radius=0.02, sphere_color=(0.2,0.8,1.0,1.0), add_color_gradient=True, inner_r=0.01, outer_r=0.03, hit_rgba=np.array([255, 64, 32, 255], np.uint8), gamma=2.5)
            
            self.proximity_tracker.update(mesh.probe_centroid, mesh.smpl_joints[9:15, :])

            self.viewer.viewer_flags['caption'][0]['text'] = f"{self.scanning_progress}"

        mesh_pyrender = pyrender.Mesh.from_trimesh(self.mesh.mesh_scene)
        self.mn = pyrender.Node(mesh=mesh_pyrender)
        self.an = pyrender.Node(mesh=arrow_pyrender)
        self.scene.add_node(self.mn)

        self.viewer.render_lock.release()

        self.frame_count += 1


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

    def add_custom_marker_az(self,
                            joints,                # (N,3) numpy
                            joint_id_a,
                            joint_id_b,
                            l,               # metres from A along AB
                            alpha_deg,       # azimuth inside X-Z plane
                            probe_centroid,  # (1,3) x,y,z cooordinates of probe centroid
                            add_shpere = False,
                            sphere_radius=0.02,
                            sphere_color=(0.2,0.8,1.0,1.0)
                            ,add_color_gradient = True
                            ,inner_r  = 0.01
                            ,outer_r = 0.03
                            ,hit_rgba = np.array([255, 64, 32, 255], np.uint8)
                            ,gamma = 5):
        """
        l is length along the line from joint A to joint B
        α is measured in the plane perpendicular to AB:
            0° → local +X
        90° → local +Z
        180° → local −X      (-y is impossible because the whole plane is ⟂ y)
        gamma  # >1 steeper, <1 gentler, 1 = linear gamma controls the gradient

        # refer to JOINT_NAMES in the common/preprocessing_operations for joint numbers
        
        THis stuff controls the sphere addition, radius and color
        add_shpere = False,
        sphere_radius=0.02,
        sphere_color=(0.2,0.8,1.0,1.0)
            
        The below stuff controls the color gradient and its color
            add_color_gradient = True
        ,inner_r  = 0.01
        ,outer_r = 0.03
        ,hit_rgba = np.array([255, 64, 32, 255], np.uint8)
        ,gamma = 2.5)
        Returns hit-point or None.
        """
        print("add_custom_marker_az")
        print("joint_id_a", joint_id_a)
        print("joint_id_b", joint_id_b)
        print(joints[0])
        print(joints[joint_id_a])
        print(joints[joint_id_b])
        pA, pB = joints[joint_id_a], joints[joint_id_b]
        vAB    = pB - pA
        normAB = np.linalg.norm(vAB)
        if normAB < 1e-8:
            raise ValueError("AB length ≈ 0; cannot build frame.")

        # local frame ---------------------------------------------------
        y_axis = vAB / normAB                                            # +Y
        # choose an arbitrary world-up that is *not* colinear with y_axis
        world_up = np.array([0, 0, 1.0])
        if abs(np.dot(world_up, y_axis)) > 0.95:                         # almost colinear
            world_up = np.array([0, 1.0, 0])
        x_axis = np.cross(world_up, y_axis)
        x_axis /= np.linalg.norm(x_axis)                                 # +X
        z_axis = np.cross(y_axis, x_axis)                                # +Z (right-handed)

        # let's do it based on probe centroid - Jack123
        probe_distance, AB_intersection_point = self.point_to_line_distance_and_intersection_3d(probe_centroid, pA, pB)
        print(f"probe_distance: {probe_distance}")
        distance_threshold = 0.2
        if self.is_point_on_line_segment(AB_intersection_point, pA, pB) and probe_distance < distance_threshold:
            self.custom_markers.add(tuple(probe_centroid))
            self.scanning_progress["thigh"] += 1


        origin = AB_intersection_point
        # ray origin & direction ---------------------------------------
        #origin   = pA + (l / normAB) * vAB                               # on AB
        alpha    = math.radians(alpha_deg)
        dir_vec  =  math.cos(alpha) * x_axis + math.sin(alpha) * z_axis
        dir_vec /= np.linalg.norm(dir_vec)

        # ray-mesh intersection ----------------------------------------
        loc, *_ = self.mesh.mesh_scene.ray.intersects_location(
                    origin.reshape(1,3), dir_vec.reshape(1,3),
                    multiple_hits=False)
        if len(loc)==0:
            return None
        hit = loc[0]
        probe_squared_distance_to_hit = np.sum((hit - probe_centroid) ** 2, axis=0)
        print(f"probe_squared_distance_to_hit: {probe_squared_distance_to_hit}")
        radius = 0.3
        close = probe_squared_distance_to_hit < radius ** 2



            # ───────────────── colour-gradient around the hit ─────────────────
        if add_color_gradient:
                    
            dists   = np.linalg.norm(self.mesh.mesh_scene.vertices - hit, axis=1)
            in_band = dists < outer_r
            
            if np.any(in_band):
                                
                lin   = (outer_r - dists[in_band]) / (outer_r - inner_r)
                w     = np.clip(lin, 0.0, 1.0) ** gamma
                w     = w[:, None]                                  # (k,1)
            
                base  = self.mesh.mesh_scene.visual.vertex_colors[in_band].astype(np.float32)
                target = hit_rgba.astype(np.float32)                # make it float for math
                blend = (w * target + (1.0 - w) * base).astype(np.uint8)
            
                self.mesh.mesh_scene.visual.vertex_colors[in_band] = blend
                #self.scene.add(mesh_trimesh)
        
        for pc in self.custom_markers:
            pc = np.array(pc)
            _, ABip = self.point_to_line_distance_and_intersection_3d(pc, pA, pB)
            origin = ABip
            # ray origin & direction ---------------------------------------
            #origin   = pA + (l / normAB) * vAB                               # on AB
            alpha    = math.radians(alpha_deg)
            dir_vec  =  math.cos(alpha) * x_axis + math.sin(alpha) * z_axis
            dir_vec /= np.linalg.norm(dir_vec)

            # ray-mesh intersection ----------------------------------------
            loc, *_ = self.mesh.mesh_scene.ray.intersects_location(
                        origin.reshape(1,3), dir_vec.reshape(1,3),
                        multiple_hits=False)
            if len(loc)==0:
                return None
            hit = loc[0]

            dists   = np.linalg.norm(self.mesh.mesh_scene.vertices - hit, axis=1)
            in_band = dists < outer_r
            
            if np.any(in_band):
                                
                lin   = (outer_r - dists[in_band]) / (outer_r - inner_r)
                w     = np.clip(lin, 0.0, 1.0) ** gamma
                w     = w[:, None]                                  # (k,1)
            
                base  = self.mesh.mesh_scene.visual.vertex_colors[in_band].astype(np.float32)
                target = hit_rgba.astype(np.float32)                # make it float for math
                blend = (w * target + (1.0 - w) * base).astype(np.uint8)
            
                self.mesh.mesh_scene.visual.vertex_colors[in_band] = blend
                #self.scene.add(mesh_trimesh)

        # visual sphere -------------------------------------------------
        if add_shpere and self.is_point_on_line_segment(AB_intersection_point, pA, pB) and probe_distance < distance_threshold:
            sph = trimesh.creation.uv_sphere(radius=sphere_radius)
            sph.visual.vertex_colors = sphere_color
            sph.apply_translation(hit)
            cm = self.scene.add(pyrender.Mesh.from_trimesh(sph))
            self.custom_markers.add(cm)
        return hit
    

    def point_to_line_distance_and_intersection_3d(self, point, line_point1, line_point2):
        """
        Calculate the perpendicular distance of a point from a line in 3D space
        and return the point of intersection on the line.

        Parameters:
            point (np.array): The point in 3D space (3D coordinates).
            line_point1 (np.array): A point on the line (3D coordinates).
            line_point2 (np.array): Another point on the line (3D coordinates).

        Returns:
            tuple: A tuple containing:
                - float: The perpendicular distance from the point to the line.
                - np.array: The point of intersection on the line (3D coordinates).
        """
        point = np.array(point)
        line_point1 = np.array(line_point1)
        line_point2 = np.array(line_point2)

        # Direction vector of the line
        line_direction = line_point2 - line_point1
        line_direction_normalized = line_direction / np.linalg.norm(line_direction)

        # Vector from line_point1 to the given point
        point_vector = point - line_point1

        # Projection of the point vector onto the line direction
        projection_length = np.dot(point_vector, line_direction_normalized)
        intersection_point = line_point1 + projection_length * line_direction_normalized

        # Distance is the norm of the vector from the point to the intersection point
        distance = np.linalg.norm(point - intersection_point)

        return distance, intersection_point
    
    def is_point_on_line_segment(self, point, line_start, line_end, tolerance=1e-6):
        """
        Check if a point lies on a line segment in 3D space.

        Parameters:
            point (numpy.ndarray): The point to check (3D coordinates).
            line_start (numpy.ndarray): The start point of the line segment (3D coordinates).
            line_end (numpy.ndarray): The end point of the line segment (3D coordinates).
            tolerance (float): A small tolerance value to account for floating-point errors.

        Returns:
            bool: True if the point lies on the line segment, False otherwise.
        """
        # Convert inputs to numpy arrays
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)

        # Check if the point is collinear with the line segment
        line_vector = line_end - line_start
        point_vector = point - line_start

        # Compute the cross product to check collinearity
        cross_product = np.cross(line_vector, point_vector)
        if not np.allclose(cross_product, 0, atol=tolerance):
            return False

        # Check if the point lies within the bounds of the line segment
        dot_product = np.dot(point_vector, line_vector)
        if dot_product < 0 or dot_product > np.dot(line_vector, line_vector):
            return False

        return True



    # def point_to_line_distance_3d(self, point, line_point1, line_point2):
    #     """
    #     Calculate the perpendicular distance of a point from a line in 3D space.

    #     Parameters:
    #         point (np.array): The point in 3D space (3D coordinates).
    #         line_point1 (np.array): A point on the line (3D coordinates).
    #         line_point2 (np.array): Another point on the line (3D coordinates).

    #     Returns:
    #         float: The perpendicular distance from the point to the line.
    #     """
    #     point = np.array(point)
    #     line_point1 = np.array(line_point1)
    #     line_point2 = np.array(line_point2)

    #     # Direction vector of the line
    #     line_direction = line_point2 - line_point1

    #     # Vector from line_point1 to the given point
    #     point_vector = point - line_point1

    #     # Cross product of the direction vector and the point vector
    #     cross_product = np.cross(line_direction, point_vector)

    #     # Magnitude of the cross product
    #     cross_magnitude = np.linalg.norm(cross_product)

    #     # Magnitude of the direction vector
    #     line_magnitude = np.linalg.norm(line_direction)

    #     # Distance is the ratio of the magnitudes
    #     distance = cross_magnitude / line_magnitude

    #     return distance
