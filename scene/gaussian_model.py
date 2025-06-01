#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # Attributes for managing multiple named segmentation masks
        self.named_segment_masks = {}  # Stores label_name: mask_tensor (full cloud)
        self.segment_label_ids = {}    # Stores label_name: integer_id
        self.next_label_id = 1         # Next available ID for new labels

        self.old_xyz = []
        self.old_mask = []

        self.old_features_dc = []
        self.old_features_rest = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_rotation = []

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._mask,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_mask(self):
        return self._mask
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        mask = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self._mask = nn.Parameter(mask.requires_grad_(True))
        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},

            # {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('label_id') # Add the new attribute here
        return l

    def save_ply(self, path, mask: torch.Tensor = None, segment_label: str = None):
        mkdir_p(os.path.dirname(path))

        # Detach all relevant tensors first
        current_xyz = self._xyz.detach()
        current_features_dc = self._features_dc.detach()
        current_features_rest = self._features_rest.detach()
        current_opacities = self._opacity.detach()
        current_scaling = self._scaling.detach()
        current_rotation = self._rotation.detach()

        # Initialize tensors to be saved with current (possibly already segmented) data
        xyz_to_save = current_xyz
        features_dc_to_save = current_features_dc
        features_rest_to_save = current_features_rest
        opacities_to_save = current_opacities
        scaling_to_save = current_scaling
        rotation_to_save = current_rotation

        if mask is not None:
            processed_mask = mask.squeeze().bool()

            if processed_mask.shape[0] != current_xyz.shape[0]:
                print(f"Warning/Error in save_ply: Provided mask shape {processed_mask.shape} does not match current XYZ shape {current_xyz.shape}.")
                if self.segment_times > 0:
                    print("Info: Model is already segmented (segment_times > 0). Provided mask is incompatible and will be IGNORED. Saving current (segmented) state of the model.")
                    # No change to xyz_to_save etc., they remain the current segmented state.
                else:
                    # segment_times == 0, model is original/unsegmented. Mask should have matched.
                    print("Error: Mask shape mismatch for an unsegmented model. Cannot save PLY.")
                    return  # Critical error, cannot proceed
            else: # Mask shape is compatible with current_xyz
                if self.segment_times > 0:
                    # This case implies the provided mask *matches* the already segmented data.
                    # This could be an explicit request to further filter the *already segmented* data.
                    print("Info: Model is already segmented, and a compatible mask was provided. Applying this mask to the current (segmented) data.")
                    xyz_to_save = current_xyz[processed_mask]
                    features_dc_to_save = current_features_dc[processed_mask]
                    features_rest_to_save = current_features_rest[processed_mask]
                    opacities_to_save = current_opacities[processed_mask]
                    scaling_to_save = current_scaling[processed_mask]
                    rotation_to_save = current_rotation[processed_mask]
                else: # segment_times == 0, mask is compatible with original data
                    print("Info: Applying provided mask to unsegmented model data.")
                    xyz_to_save = current_xyz[processed_mask]
                    features_dc_to_save = current_features_dc[processed_mask]
                    features_rest_to_save = current_features_rest[processed_mask]
                    opacities_to_save = current_opacities[processed_mask]
                    scaling_to_save = current_scaling[processed_mask]
                    rotation_to_save = current_rotation[processed_mask]
        else: # mask is None
            print("Info: No mask provided. Saving current state of the model.")
            # xyz_to_save etc. are already set to current values

        # Convert to NumPy arrays for saving
        xyz_np = xyz_to_save.cpu().numpy()

        if xyz_np.shape[0] == 0:
            print(f"Warning: No points to save for {path} (possibly due to mask or empty model). PLY file will not be written or will be empty.")
            # Optionally, write an empty PLY or just return
            # PlyData([]).write(path) # To write an empty valid PLY
            return

        normals_np = np.zeros_like(xyz_np)
        f_dc_np = features_dc_to_save.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_np = features_rest_to_save.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities_np = opacities_to_save.cpu().numpy()
        scale_np = scaling_to_save.cpu().numpy()
        rotation_np = rotation_to_save.cpu().numpy()

        # Create label_id_np array
        label_id_value = 0  # Default value
        if segment_label and segment_label.strip(): # If a non-empty label string is given
            label_id_value = 1 # Assign 1 if any label is present
        label_id_np = np.full((xyz_np.shape[0], 1), label_id_value, dtype=np.uint8)

        # Revised dtype_full construction
        attributes_list = self.construct_list_of_attributes()
        dtype_full = []
        for attribute_name in attributes_list:
            if attribute_name == 'label_id':
                dtype_full.append((attribute_name, 'u1'))
            else:
                dtype_full.append((attribute_name, 'f4')) # Default for others

        elements = np.empty(xyz_np.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz_np, normals_np, f_dc_np, f_rest_np, opacities_np, scale_np, rotation_np, label_id_np), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    #     # edit
    #     add_color = True
    #     if add_color:
    #         dtype_full[3], dtype_full[4], dtype_full[5] = ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    #         rgbs = SH2RGB(f_dc)
    #         normals = (np.clip(rgbs, 0.0, 1.0) * 255).astype(np.uint8)
            
    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def save_mask(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()        
        np.save(path, mask)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # mask = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        # self._mask = nn.Parameter(torch.tensor(mask, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @torch.no_grad()
    def segment(self, mask=None):
        """
        This method is now a NO-OP regarding model geometry/attribute pruning.
        It was previously used to apply a mask to self._xyz and related features,
        effectively segmenting the model by removing points.
        This functionality is removed to ensure self._xyz always represents the
        original, complete point cloud. Named segmentations are handled by
        self.named_segment_masks and add_or_update_segment_label.
        """
        # The following lines are removed/commented out:
        # - Appending to self.old_xyz, self.old_features_dc, etc.
        # - Modification of self._xyz, self._features_dc, etc. based on the mask.
        # - Calls to self._prune_optimizer that would modify these attributes.
        # - Updates to self.segment_times and self._mask related to this pruning.
        
        # if mask is not None:
        #     print(f"Debug: GaussianModel.segment called with mask. Shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
        # else:
        #     print("Debug: GaussianModel.segment called without mask.")
        # print("Info: GaussianModel.segment is now a no-op for geometry pruning.")
        pass

    def roll_back(self):
        """
        This method is now a NO-OP.
        It previously restored self._xyz and other attributes from self.old_xyz etc.,
        and decremented self.segment_times. Since self._xyz is no longer pruned by segment(),
        this restoration is not needed. Clearing of GUI-level selection state
        is handled in the GUI's roll_back callback.
        """
        # The following lines are removed/commented out:
        # - Popping from self.old_xyz, self.old_features_dc, etc. and assigning to self._xyz etc.
        # - Decrementing self.segment_times and related self._mask manipulation.
        # print("Info: GaussianModel.roll_back is now a no-op.")
        pass

    @torch.no_grad()
    def clear_segment(self):
        """
        This method is now simplified.
        It previously restored self._xyz to its initial state from self.old_xyz[0]
        and reset various lists and self.segment_times.
        Since self._xyz is no longer changed by segment(), it now only resets
        self.segment_times. The self._mask attribute's role is also diminished.
        Actual clearing of named segmentations would need a different method.
        """
        # The following lines are removed/commented out:
        # - Restoring self._xyz etc. from self.old_xyz[0] etc.
        # - Clearing self.old_xyz, self.old_features_dc, etc. lists.
        # - Re-initializing self._mask to ones. (self._mask's role is reduced)

        self.segment_times = 0 # Reset counter, though its primary use was with pruning.
        # print("Info: GaussianModel.clear_segment now primarily resets segment_times.")
        pass

    def add_or_update_segment_label(self, label_name: str, selection_mask: torch.Tensor):
        """
        Adds a new segment label and its mask, or updates the mask for an existing label.

        Args:
            label_name (str): The user-defined name for the segment.
            selection_mask (torch.Tensor): A boolean tensor mask aligned with the
                                           original full point cloud. True values indicate
                                           points belonging to this segment label.
        """
        if not isinstance(label_name, str) or not label_name.strip():
            print("Error: Segment label name cannot be empty.")
            return

        if not isinstance(selection_mask, torch.Tensor) or selection_mask.dtype != torch.bool:
            print(f"Error: selection_mask for label '{label_name}' must be a boolean torch.Tensor.")
            return

        # Placeholder for a more robust size check if needed in the future,
        # e.g., if self._xyz could be a pruned version of the original cloud.
        # For now, this method assumes selection_mask is aligned with the original point cloud.
        # if hasattr(self, 'initial_point_count') and selection_mask.shape[0] != self.initial_point_count:
        #     print(f"Error: selection_mask shape {selection_mask.shape} does not match initial point cloud size {self.initial_point_count}.")
        #     return

        self.named_segment_masks[label_name] = selection_mask.clone() # Store a clone

        if label_name not in self.segment_label_ids:
            current_id = self.next_label_id
            self.segment_label_ids[label_name] = current_id
            self.next_label_id += 1
            print(f"Added new segment label: '{label_name}' with ID {current_id}")
        else:
            # Mask is updated, ID remains the same.
            print(f"Updated mask for segment label: '{label_name}' (ID: {self.segment_label_ids[label_name]})")

    def save_ply_with_all_labels(self, path: str):
        """
        Saves the entire original point cloud with a 'classification_id' scalar field.
        The ID for each point is determined by the named segment masks.
        """
        mkdir_p(os.path.dirname(path))

        # --- Data Preparation ---
        # These are assumed to be the original, full point cloud attributes.
        xyz_np = self._xyz.detach().cpu().numpy()
        # No normals are explicitly stored in GaussianModel for PLY, so create zeros.
        # Or, if there's a way to get original normals, use that. For now, zeros.
        normals_np = np.zeros_like(xyz_np)
        f_dc_np = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_np = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities_np = self._opacity.detach().cpu().numpy() # These are raw opacities
        scale_np = self._scaling.detach().cpu().numpy()     # These are raw scales
        rotation_np = self._rotation.detach().cpu().numpy()

        if xyz_np.shape[0] == 0:
            print(f"Warning: Original point cloud is empty. Cannot save {path}.")
            return

        # --- Classification ID Calculation ---
        num_points = xyz_np.shape[0]
        # Initialize classification_ids with 0 (unlabeled)
        classification_id_np = np.zeros((num_points, 1), dtype=np.uint16) # Using uint16 for more IDs if needed

        # Iterate through named segments. Handle overlaps by priority (e.g., first found).
        # For more sophisticated overlap handling, a different approach would be needed.
        # Consider sorting labels by ID or name if consistent overwrite behavior is desired.
        sorted_label_names = sorted(self.segment_label_ids.keys(), key=lambda k: self.segment_label_ids[k])

        for label_name in sorted_label_names:
            if label_name in self.named_segment_masks:
                mask_tensor = self.named_segment_masks[label_name]
                if mask_tensor.shape[0] != num_points:
                    print(f"Warning: Mask for label '{label_name}' has shape {mask_tensor.shape} "
                          f"which does not match point cloud size {num_points}. Skipping this label.")
                    continue

                label_id = self.segment_label_ids[label_name]
                # Apply mask: points where mask is True get this label_id
                # This will overwrite if a point is in multiple masks; last one processed by this loop wins
                # if not sorted, or first one if sorted and we check if classification_id_np is 0.
                # To ensure first label found wins for a point:
                # classification_id_np[mask_tensor.cpu().numpy() & (classification_id_np == 0)] = label_id
                # For "last label applied wins" (simpler):
                classification_id_np[mask_tensor.cpu().numpy()] = label_id


        # --- PLY Structure Definition ---
        # Get base attributes list (excluding the new classification_id)
        # temp_model = GaussianModel(self.max_sh_degree) # Create a temp instance to call construct_list_of_attributes
                                                     # This is a bit of a hack. Better if construct_list_of_attributes
                                                     # was static or didn't depend on features_dc etc. being populated.
                                                     # Or, modify construct_list_of_attributes to be more flexible.
                                                     # For now, this will work if __init__ doesn't crash it.
        
        # Let's define the attribute list directly for this save function to avoid issues with
        # construct_list_of_attributes() if it relies on specific states of features_dc etc.
        # or the 'label_id' from the *other* save function.

        attributes_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # Ensure features_dc and features_rest are on CPU and detached for shape access if not already
        features_dc_shape = self._features_dc.shape
        features_rest_shape = self._features_rest.shape

        for i in range(features_dc_shape[1]*features_dc_shape[2]):
            attributes_names.append(f'f_dc_{i}')
        for i in range(features_rest_shape[1]*features_rest_shape[2]):
            attributes_names.append(f'f_rest_{i}')
        attributes_names.append('opacity')
        for i in range(self._scaling.shape[1]):
            attributes_names.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            attributes_names.append(f'rot_{i}')
        # Add our new classification field
        attributes_names.append('classification_id')

        dtype_full = []
        for attribute_name in attributes_names:
            if attribute_name == 'classification_id':
                dtype_full.append((attribute_name, 'u2')) # uint16, same as classification_id_np
            else:
                dtype_full.append((attribute_name, 'f4')) # Default for others

        # --- Concatenate all attributes for PLY ---
        all_attributes_np = np.concatenate(
            (xyz_np, normals_np, f_dc_np, f_rest_np, opacities_np, scale_np, rotation_np, classification_id_np),
            axis=1
        )

        elements = np.empty(num_points, dtype=dtype_full)
        elements[:] = list(map(tuple, all_attributes_np))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Saved multi-label PLY to {path} with 'classification_id' field.")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        # "mask": new_mask,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # new_mask = self._mask[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        # new_mask = self._mask[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1