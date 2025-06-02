# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    FEATURE_DIM = 32
    MODEL_PATH = './output/figurines' # 30000

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate
        }

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)
        self.seg_score = None

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
        self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        self.cluster_in_3D_flag = False

        self.render_mode_rgb = False
        self.render_mode_similarity = False
        self.render_mode_pca = False
        self.render_mode_cluster = False

        self.save_flag = False
        self.active_selection_mask = None # New attribute for storing current selection mask
        self.show_3d_preview = False      # Boolean to control 3D preview state
        self.preview_selection_mask = None # Stores the point-based mask for 3D preview

    def callback_save_selection_to_label(self, sender=None, app_data=None):
        print("DEBUG: callback_save_selection_to_label called.")
        label_name = dpg.get_value("_new_label_name_input")

        # 1. Error Handling for Empty Label Name
        if not label_name.strip():
            error_modal_tag = "empty_label_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Label name cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Error Handling for No Active Selection
        if self.active_selection_mask is None:
            error_modal_tag = "no_active_selection_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text("No active selection to save.\nPlease use 'segment3d' first to create a selection.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Mask Usage (direct use of self.active_selection_mask)
        # CRITICAL ASSUMPTION: self.active_selection_mask is aligned with the original full point cloud.

        # Optional: Add a check for mask shape if self.engine['scene']._xyz is accessible
        # and represents the original cloud. Given the refactor ensuring _xyz is original, this should hold.
        if self.active_selection_mask.shape[0] != self.engine['scene'].get_xyz.shape[0]:
            warning_modal_tag = "mask_shape_warning_modal"
            print(f"Warning: Active selection mask shape {self.active_selection_mask.shape} "
                  f"does not match scene XYZ shape {self.engine['scene'].get_xyz.shape[0]}. This could indicate an issue.")
            if dpg.does_item_exist(warning_modal_tag):
                dpg.delete_item(warning_modal_tag)
            with dpg.window(modal=True, label="Warning", tag=warning_modal_tag, width=450, height=120, no_close=True) as modal_id:
                dpg.add_text(f"Mask shape mismatch:\nSelection: {self.active_selection_mask.shape[0]}, Scene: {self.engine['scene'].get_xyz.shape[0]}.\nThis might indicate an internal issue if the selection\nwas not derived from the current full point cloud.\nProceed with caution or clear selection and retry.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            # Depending on severity, might 'return' here. For now, it's a warning.

        try:
            # 4. Call to Backend
            self.engine['scene'].add_or_update_segment_label(label_name, self.active_selection_mask)

            # 5. GUI Updates After Successful Save
            if hasattr(self.engine['scene'], 'segment_label_ids'):
                dpg.configure_item("_labels_listbox", items=list(self.engine['scene'].segment_label_ids.keys()))

            dpg.set_value("_new_label_name_input", "") # Clear input field

            # Clear current selection state
            self.active_selection_mask = None
            self.new_click_xy = []  # Clear click prompts
            self.prompt_num = 0

            # Also reset 3D preview state after saving a label
            self.show_3d_preview = False
            self.preview_selection_mask = None
            self.update_camera = True # Ensure view refreshes
            print(f"Segment label '{label_name}' saved/updated successfully. Selection and 3D preview cleared.")

        except AttributeError as e:
            # 6. Error Handling for Backend Call (AttributeError)
            error_modal_tag = "attr_error_modal"
            print(f"AttributeError: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n'add_or_update_segment_label' method not found\non scene object. Details: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            # 6. Error Handling for Backend Call (General Exception)
            error_modal_tag = "general_save_label_error_modal"
            print(f"Error saving label: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred while saving the label:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_export_all_labels_to_ply(self, sender=None, app_data=None):
        print("DEBUG: callback_export_all_labels_to_ply called.")
        base_filename = dpg.get_value("_export_all_labels_filename_input")

        # 1. Validate Filename
        if not base_filename.strip():
            error_modal_tag = "empty_export_filename_error_modal"
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                dpg.add_text("Export filename base cannot be empty.")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 2. Ensure output directory exists
        output_dir = "./segmentation_res"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            error_modal_tag = "export_dir_error_modal"
            print(f"OSError creating directory {output_dir}: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Could not create directory:\n{output_dir}\nError: {str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
            return

        # 3. Construct Full Path
        filepath = os.path.join(output_dir, base_filename + ".ply")

        # 4. Call Backend Method
        try:
            if not hasattr(self.engine['scene'], 'save_ply_with_all_labels'):
                raise AttributeError("'save_ply_with_all_labels' method not found on scene object.")

            self.engine['scene'].save_ply_with_all_labels(filepath)

            # Success Message
            success_modal_tag = "export_all_success_modal"
            print(f"Successfully exported all labels to {filepath}")
            if dpg.does_item_exist(success_modal_tag):
                dpg.delete_item(success_modal_tag)
            with dpg.window(modal=True, label="Success", tag=success_modal_tag, width=450, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Successfully exported all labels to:\n{filepath}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

        except AttributeError as e:
            error_modal_tag = "export_attr_error_modal"
            print(f"AttributeError during export: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"Internal Error:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        except Exception as e:
            error_modal_tag = "general_export_error_modal"
            print(f"Error during export all labels: {e}")
            if dpg.does_item_exist(error_modal_tag):
                dpg.delete_item(error_modal_tag)
            with dpg.window(modal=True, label="Error", tag=error_modal_tag, width=400, height=100, no_close=True) as modal_id:
                dpg.add_text(f"An error occurred during export:\n{str(e)}")
                dpg.add_spacer(height=5)
                dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))

    def callback_toggle_3d_preview(self, sender=None, app_data=None):
        self.show_3d_preview = not self.show_3d_preview
        if self.show_3d_preview:
            # If there's an active selection mask (from a finalized "segment3d" action), use that.
            if self.active_selection_mask is not None:
                self.preview_selection_mask = self.active_selection_mask
                print("DEBUG: 3D Preview ON. Using active_selection_mask.")
            # Else, if there's a fresh point-based score from clicks (score_pts_binary), use that.
            # This allows previewing before finalizing with "segment3d".
            elif hasattr(self, 'score_pts_binary') and self.score_pts_binary is not None:
                self.preview_selection_mask = self.score_pts_binary
                print("DEBUG: 3D Preview ON. Using current score_pts_binary from clicks.")
            else:
                # No selection available to preview. Turn preview off.
                self.show_3d_preview = False
                self.preview_selection_mask = None
                print("DEBUG: No selection (active_selection_mask or score_pts_binary) available for 3D Preview. Preview remains OFF.")
                # Optionally, show a DPG modal error/info message here
                error_modal_tag = "no_selection_for_preview_modal"
                if dpg.does_item_exist(error_modal_tag):
                    dpg.delete_item(error_modal_tag)
                with dpg.window(modal=True, label="Info", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                    dpg.add_text("No 3D selection available to preview.\nUse clicks and 'segment3d' first.")
                    dpg.add_spacer(height=5)
                    dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
        else:
            self.preview_selection_mask = None
            print("DEBUG: 3D Preview OFF.")

        # Trigger a re-render if needed (update_camera usually does this)
        self.update_camera = True


    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def preview_callback(sender):
            self.preview = dpg.get_value(sender)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_reload():
            self.reload_flag = True
        def callback_cluster():
            self.cluster_in_3D_flag =True
        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            try:
                self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
                self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)
            except:
                pass

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        def render_mode_similarity_callback(sender):
            self.render_mode_similarity = not self.render_mode_similarity
        def render_mode_pca_callback(sender):
            self.render_mode_pca = not self.render_mode_pca
        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = not self.render_mode_cluster
        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_slider_float(label="Scale", default_value=0.5,
                                 min_value=0.0, max_value=1.0, tag="_Scale")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            # dpg.add_button(label="render_option", tag="_button_depth",
                            # callback=callback_depth)
            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_checkbox(label="RGB", callback=render_mode_rgb_callback, user_data="Some Data")
            dpg.add_checkbox(label="PCA", callback=render_mode_pca_callback, user_data="Some Data")
            dpg.add_checkbox(label="SIMILARITY", callback=render_mode_similarity_callback, user_data="Some Data")
            dpg.add_checkbox(label="3D CLUSTER", callback=render_mode_cluster_callback, user_data="Some Data")
            

            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="preview_segmentation_in_2d", callback=preview_callback, user_data="Some Data")
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            # New Preview Button
            dpg.add_button(label="Preview 3D Selection", tag="_preview_3d_button", callback=self.callback_toggle_3d_preview)
            dpg.add_button(label="roll_back", callback=roll_back, user_data="Some Data")
            dpg.add_button(label="Clear Click Prompts", callback=clear_edit, user_data="Some Data") # Renamed "clear" button
            dpg.add_button(label="save as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="Save Filename Base", default_value="precomputed_mask", tag="save_name", hint="Filename without extension")
            dpg.add_input_text(label="Segment Label (for PLY)", default_value="", tag="_segment_label_input", hint="E.g., window_sill (optional)")

            dpg.add_separator()
            dpg.add_text("Segment Label Management")
            dpg.add_input_text(label="New Label Name", tag="_new_label_name_input", default_value="", hint="Enter label for current selection")
            dpg.add_button(label="Save Current Selection to Label", callback=self.callback_save_selection_to_label)
            dpg.add_listbox(tag="_labels_listbox", items=[], label="Saved Segment Labels", num_items=3)

            dpg.add_separator()
            dpg.add_text("Export All Labeled Data")
            dpg.add_input_text(label="Export All Filename Base", default_value="scene_with_all_labels", tag="_export_all_labels_filename_input", hint="Filename without .ply extension")
            dpg.add_button(label="Export All Labels to PLY", callback=self.callback_export_all_labels_to_ply)
            dpg.add_text("\n")

            dpg.add_button(label="cluster3d", callback=callback_cluster, user_data="Some Data")
            dpg.add_button(label="reshuffle_cluster_color", callback=callback_reshuffle_color, user_data="Some Data")
            dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")

            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving


        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle


        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def cluster_in_3D(self):
        # try:
        #     self.engine['scene'].roll_back()
        #     self.engine['feature'].roll_back()
        # except:
        #     pass
        point_features = self.engine['feature'].get_point_features

        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates.unsqueeze(0)

        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)

        sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]

        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)

        clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)

        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())

        cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
        for i in range(0, len(np.unique(cluster_labels))):
            cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

        self.seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
        self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
        # self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)


    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16, 
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        
        return proj_mat
    

    def do_pca(self):
        sems = self.engine['feature'].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)
        print("project mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        scene_to_render = self.engine['scene']
        feature_model_to_render = self.engine['feature']
        is_previewing_isolate = False

        if self.show_3d_preview and self.preview_selection_mask is not None:
            original_scene = self.engine['scene']
            num_original_points = original_scene.get_xyz.shape[0]

            if not (isinstance(self.preview_selection_mask, torch.Tensor) and self.preview_selection_mask.dtype == torch.bool):
                print("Warning: preview_selection_mask is not a boolean tensor. Turning off 3D preview.")
                self.show_3d_preview = False
                self.preview_selection_mask = None
            elif self.preview_selection_mask.shape[0] != num_original_points:
                print(f"Warning: preview_selection_mask shape {self.preview_selection_mask.shape} "
                      f"does not match scene XYZ shape {num_original_points}. Turning off 3D preview.")
                self.show_3d_preview = False
                self.preview_selection_mask = None
            else:
                num_selected_points = torch.count_nonzero(self.preview_selection_mask).item()
                if num_selected_points == 0:
                    print("Warning: Preview selection is empty. Turning off 3D preview.")
                    self.show_3d_preview = False
                    self.preview_selection_mask = None
                    # Optionally show a DPG modal here
                    error_modal_tag = "empty_preview_selection_modal"
                    if dpg.does_item_exist(error_modal_tag):
                        dpg.delete_item(error_modal_tag)
                    with dpg.window(modal=True, label="Info", tag=error_modal_tag, width=350, height=100, no_close=True) as modal_id:
                        dpg.add_text("The current selection is empty.\nNothing to preview in isolate mode.")
                        dpg.add_spacer(height=5)
                        dpg.add_button(label="Ok", width=-1, callback=lambda: dpg.delete_item(modal_id))
                else:
                    print(f"DEBUG: Creating temporary model for isolate preview. Selected points: {num_selected_points}")
                    # Create and populate preview_scene_model
                    preview_scene_model = GaussianModel(sh_degree=original_scene.max_sh_degree)
                    preview_scene_model.active_sh_degree = original_scene.active_sh_degree

                    attributes_to_filter_scene = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']
                    for attr_name in attributes_to_filter_scene:
                        original_attr_tensor = getattr(original_scene, attr_name)
                        if original_attr_tensor is not None and original_attr_tensor.shape[0] == num_original_points:
                            filtered_tensor = original_attr_tensor.detach()[self.preview_selection_mask]
                            setattr(preview_scene_model, attr_name, torch.nn.Parameter(filtered_tensor))
                        elif original_attr_tensor is not None: # Handle cases like max_radii2D if they were class members
                             print(f"Warning: Attribute {attr_name} shape mismatch or not handled for preview model, skipping.")

                    # Critical: Reinitialize optimizer-dependent states if they are used by getters or rendering
                    # For simple preview, this might not be strictly necessary if render() directly uses the nn.Parameters
                    # preview_scene_model.max_radii2D might need to be re-calculated or filtered.
                    # However, for basic rendering, the above attributes are key.
                    # Let's assume max_radii2D isn't directly used by basic render or will work with fewer points.

                    scene_to_render = preview_scene_model
                    is_previewing_isolate = True

                    # Handle FeatureGaussianModel similarly
                    original_feature_model = self.engine['feature']
                    if hasattr(original_feature_model, '_xyz'): # Check if it's point-based
                        preview_feature_model = FeatureGaussianModel(original_feature_model.feature_dim) # Assuming feature_dim attr
                        preview_feature_model.active_sh_degree = original_feature_model.active_sh_degree # if exists

                        attributes_to_filter_feature = ['_xyz', '_point_features', '_opacity', '_scaling', '_rotation'] # Adjust based on FeatureGaussianModel
                        for attr_name in attributes_to_filter_feature:
                            if hasattr(original_feature_model, attr_name):
                                original_attr_tensor = getattr(original_feature_model, attr_name)
                                if original_attr_tensor is not None and original_attr_tensor.shape[0] == num_original_points:
                                    filtered_tensor = original_attr_tensor.detach()[self.preview_selection_mask]
                                    setattr(preview_feature_model, attr_name, torch.nn.Parameter(filtered_tensor))
                        feature_model_to_render = preview_feature_model
                    else: # If feature model isn't point-based in the same way, use original
                        feature_model_to_render = original_feature_model


        # --- Main Rendering Calls ---
        scene_outputs = render(view_camera, scene_to_render, self.opt, self.bg_color)

        # Conditional rendering for features if it was filtered or not
        feature_outputs = render_contrastive_feature(view_camera, feature_model_to_render, self.opt, self.bg_feature)

        if self.cluster_in_3D_flag:
            self.cluster_in_3D_flag = False
            print("Clustering in 3D...")
            self.cluster_in_3D()
            print("Clustering finished.")

        # Handle self.rendered_cluster logic considering the preview
        if is_previewing_isolate:
            self.rendered_cluster = None # No cluster view when isolating
        elif self.render_mode_cluster and self.cluster_point_colors is not None:
            # Ensure cluster_point_colors is compatible with scene_to_render (which is original scene here)
            if self.cluster_point_colors.shape[0] == scene_to_render.get_xyz.shape[0]:
                 self.rendered_cluster = render(view_camera, scene_to_render, self.opt, self.bg_color, override_color=self.cluster_point_colors)["render"].permute(1, 2, 0)
            else:
                 print("Warning: cluster_point_colors shape mismatch with scene model. Disabling cluster view.")
                 self.rendered_cluster = None
                 self.render_mode_cluster = False # Turn off if problematic
        else:
            self.rendered_cluster = None

        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)
        rgb_score = img.clone() # This will be the image shown, potentially modified by 2D (pixel-based) preview below
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = feature_outputs["render"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems @ self.proj_mat
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        scale = dpg.get_value('_Scale')
        self.gates = self.engine['scale_gate'](torch.tensor([scale]).cuda())
        scale_gated_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            self.active_selection_mask = None # Clear active selection mask
            # Calls to self.engine['scene'].clear_segment() are kept as it now only resets segment_times
            try:
                self.engine['scene'].clear_segment()
                if hasattr(self.engine.get('feature'), 'clear_segment'): # Check if feature model has it
                    self.engine['feature'].clear_segment()
            except Exception as e:
                print(f"Error in clear_edit calling clear_segment: {e}")

            self.show_3d_preview = False
            self.preview_selection_mask = None
            self.update_camera = True
            print("GUI clear_edit: Cleared click prompts, active selection mask, and 3D preview state.")


        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False # Reset flag
            self.prompt_num = 0
            self.active_selection_mask = None # Clear active selection mask
            # Removed calls to self.engine['scene'].roll_back() and self.engine['feature'].roll_back()
            # as they are now no-ops for geometry and state is managed here.
            print("GUI roll_back: Cleared click prompts and active selection mask.")

        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
            self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
            self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
            self.do_pca()   # calculate self.proj_mat
            self.load_model = True

        score_map = None
        if len(self.new_click_xy) > 0:

            featmap = scale_gated_feat.reshape(H, W, -1)
            
            if self.new_click:
                xy = self.new_click_xy
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                self.prompt_num += 1
                self.new_click = False
            
            score_map = featmap @ self.chosen_feature
            # print(score_map.shape, score_map.min(), score_map.max(), "score_map_shape")

            score_map = (score_map + 1.0) / 2
            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            # Preview logic update:
            if self.preview:
                if self.active_selection_mask is not None:
                    # Ensure active_selection_mask is on the correct device and has the right shape for broadcasting
                    # Assuming active_selection_mask is a 1D boolean tensor for points,
                    # and img is HxWx3. We need to apply this to the rendered image.
                    # This part is tricky because active_selection_mask is on points, not pixels.
                    # The original 'score_binary' was pixel-based.
                    # For a direct application, active_selection_mask would need to be projected to screen space.
                    # However, the current structure uses score_binary (pixel-based) for preview.
                    # If segment3d was just pressed, score_binary (from clicks) might be stale or irrelevant.
                    # Let's assume for now that if active_selection_mask is set, the preview should
                    # reflect *that* selection. If active_selection_mask is point-based,
                    # direct multiplication with 'img' isn't right.
                    # The original preview was: img * torch.max(score_binary, dim=-1, keepdim=True).values
                    # score_binary was HxW. self.active_selection_mask (from score_pts_binary) is N.
                    # This suggests the preview mechanism itself might need a rethink if active_selection_mask
                    # is to be previewed directly without re-rendering or projecting.
                    #
                    # Given the current structure, if segment3d_flag was just processed,
                    # score_binary might still hold the values from the clicks that led to active_selection_mask.
                    # If clicks were cleared, score_binary would be None.

                    # Decision: If active_selection_mask is present, it means "segment3d" was used.
                    # The preview should ideally show this. However, direct use is hard.
                    # For now, if clicks are still active (score_binary is not None), let it use that.
                    # If clicks were cleared but active_selection_mask exists, the current preview logic
                    # would not show active_selection_mask.
                    # This part of the prompt "The current "preview" mode ... can remain the primary way to visualize the active selection"
                    # might imply that the existing score_binary (pixel-based mask from clicks) is sufficient for preview.
                    # Let's stick to modifying rgb_score based on a pixel-level mask.
                    # If active_selection_mask is set, it means a "segment3d" operation was done.
                    # The score_binary from the clicks leading to it is what was previewed.
                    # If clicks are cleared after, score_binary becomes None.
                    # A simple solution: if active_selection_mask is set, we assume the user wants to see that effect,
                    # but the current code applies a *pixel* mask.
                    # The most straightforward interpretation is to keep using score_binary for preview,
                    # and active_selection_mask is the thing that's *saved*.
                    # The prompt: "Ensure that when self.preview is active, it uses self.active_selection_mask (if available) for highlighting."
                    # This is contradictory if active_selection_mask is point-based and preview is pixel-based.
                    #
                    # Let's assume the intention is: if a segment3d operation has occurred (active_selection_mask is set),
                    # AND there are no *new* clicks (score_binary is None or from those same clicks),
                    # then the preview should reflect active_selection_mask. This is still hard without projection.
                    #
                    # Simpler interpretation: Preview always uses current click-based `score_binary`.
                    # `active_selection_mask` is just for saving.
                    # If this is the case, no change to preview logic needed other than what's there.
                    #
                    # Re-reading: "Ensure that when self.preview is active, it uses self.active_selection_mask (if available) for highlighting."
                    # This implies active_selection_mask needs to be visualized.
                    # The current score_binary is the result of featmap @ self.chosen_feature.
                    # If segment3d was pressed, self.chosen_feature was used to make self.score_pts_binary (-> self.active_selection_mask)
                    # So, score_binary is ALREADY the pixel projection of what formed active_selection_mask,
                    # as long as self.chosen_feature hasn't changed and clicks haven't been cleared.

                    # If clicks were cleared (len(self.new_click_xy) == 0), then score_binary would be None.
                    # In this case, if self.active_selection_mask is not None, we *don't* have a pixel mask for it.
                    # This is the tricky part.
                    # For now, let's prioritize live clicks for preview. If clicks are cleared, preview of active_selection_mask is lost
                    # with the current pixel-based preview mechanism.
                    if score_binary is not None:
                         rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values
                    # else: if self.active_selection_mask is not None, how to preview? This is the gap.
                    # For now, if score_binary is None, it implies no preview from clicks.
                    # If active_selection_mask is just for saving, then no change is needed here.
                    # Given the complexity, I will assume the existing preview mechanism based on `score_binary`
                    # (which is pixel-based) is what's intended for "preview", and `active_selection_mask`
                    # (point-based) is for the backend logic like saving.
                    # Thus, the original line for rgb_score in preview mode is likely fine.
                    # The main change is that self.active_selection_mask now *stores* the point-based selection.
                    # The prompt might have a slight ambiguity here.
                    # If a direct visualization of point-based self.active_selection_mask is needed,
                    # it would require rendering the scene with an override_color based on this mask,
                    # which is a much larger change than what seems implied for this step.

                    # Let's stick to the most direct interpretation: if score_binary is available, use it.
                    # This means if clicks are cleared, preview of that selection is also cleared.
                    if score_binary is not None:
                         rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values
                    else:
                        # No live click mask, and no direct way to show point-based active_selection_mask
                        # without re-rendering. So, show original image in preview mode.
                        rgb_score = img
            else:
                rgb_score = img
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine['feature'].get_point_features.squeeze()
                scale_gated_feat_pts = feat_pts * self.gates.unsqueeze(0)
                scale_gated_feat_pts = torch.nn.functional.normalize(scale_gated_feat_pts, dim = -1, p = 2)

                score_pts = scale_gated_feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                self.score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                # save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                # try:
                #     self.engine['scene'].roll_back() # Keep commented
                #     self.engine['feature'].roll_back() # Keep commented
                # except:
                #     pass
                # self.engine['scene'].segment(self.score_pts_binary) # REMOVED
                # self.engine['feature'].segment(self.score_pts_binary) # REMOVED
                self.active_selection_mask = self.score_pts_binary # Store the mask
                print(f"segment3d_flag: Set active_selection_mask. Point count: {torch.count_nonzero(self.active_selection_mask) if self.active_selection_mask is not None else 'None'}")


        if self.save_flag:
            print("Saving ...")
            self.save_flag = False # Reset flag at the beginning
            try:
                os.makedirs("./segmentation_res", exist_ok=True)

                # Define save_mask as per instruction
                save_mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1

                if self.engine['scene'].segment_times == 0:
                    if not dpg.does_item_exist("save_error_window"):
                        with dpg.window(label="Error", width=400, height=100, modal=True, show=True, tag="save_error_window", no_close=True):
                            dpg.add_text("Please segment an object first before saving.")
                            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.configure_item("save_error_window", show=False))
                    # Skip saving process
                else:
                    save_name_base = dpg.get_value('save_name')

                    # Save the mask tensor first. This mask (`save_mask`) is derived from the original
                    # point cloud state and the segmentation operation. It's used for saving the .pt file.
                    mask_filename = save_name_base + ".pt"
                    mask_filepath = os.path.join("./segmentation_res", mask_filename)
                    torch.save(save_mask, mask_filepath)
                    print(f"Mask tensor saved to {mask_filepath}")

                    # Construct PLY file path
                    ply_filename = save_name_base + ".ply"
                    ply_filepath = os.path.join("./segmentation_res", ply_filename)

                    # Since self.engine['scene'].segment_times > 0 here, the model's _xyz
                    # (and other attributes) are already segmented.
                    # Therefore, pass mask=None to save_ply to save this already segmented state.
                    segment_label_text = dpg.get_value("_segment_label_input")
                    self.engine['scene'].save_ply(ply_filepath, mask=None, segment_label=segment_label_text)
                    print(f"Segmented PLY saved to {ply_filepath} (using current model state as it's already segmented, segment label: '{segment_label_text}')")

                    # Display a success message
                    if not dpg.does_item_exist("save_success_window"):
                        with dpg.window(label="Success", width=450, height=120, modal=True, show=True, tag="save_success_window", no_close=True):
                            dpg.add_text(f"Segmented data saved successfully!\n\nMask: {mask_filepath}\nPLY: {ply_filepath}")
                            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.configure_item("save_success_window", show=False))

            except Exception as e:
                print(f"Error during saving: {str(e)}")
                if not dpg.does_item_exist("generic_save_error_window"):
                    with dpg.window(label="Save Error", width=400, height=100, modal=True, show=True, tag="generic_save_error_window", no_close=True):
                        dpg.add_text(f"An error occurred during saving: {str(e)}")
                        dpg.add_button(label="OK", width=-1, callback=lambda: dpg.configure_item("generic_save_error_window", show=False))

        self.render_buffer = None
        render_num = 0
        if self.render_mode_rgb or (not self.render_mode_pca and not self.render_mode_cluster and not self.render_mode_similarity):
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        
        if self.render_mode_pca:
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_cluster: # This is for the multi-view composition at the end
            if self.rendered_cluster is not None: # If 3D preview was active, rendered_cluster is None
                # Use the dedicated self.rendered_cluster if available (meaning no 3D selection preview was active)
                self.render_buffer = self.rendered_cluster.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + self.rendered_cluster.cpu().numpy().reshape(-1)
            else:
                # If 3D selection preview was active, rendered_cluster is None.
                # We still need to contribute to render_buffer if render_mode_cluster is true.
                # In this case, rgb_score (which includes the 3D selection preview) is the best we have.
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_similarity:
            if score_map is not None:
                self.render_buffer = self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32)
            else:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)

            render_num += 1
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('-f', '--feature_iteration', type=int, default=10000)
    parser.add_argument('-s', '--scene_iteration', type=int, default=30000)

    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCALE_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    opt.FEATURE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
    gui = GaussianSplattingGUI(opt, gs_model, feat_gs_model, scale_gate)

    gui.render()