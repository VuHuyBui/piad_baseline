# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #
import torch
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render_gaussians(
    viewpoint_camera, 
    pc: GaussianModel, 
    pipe, 
    bg_color: torch.Tensor,
    scaling_modifier=1.0, 
    override_color=None
):
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Intrinsics matrix
    fx = W / (2 * tanfovx)
    fy = H / (2 * tanfovy)
    K = torch.tensor(
        [[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]],
        device="cuda", 
        dtype=torch.float32
    ).unsqueeze(0)  # [1, 3, 3]

    # world_view_transform is W2C transposed (3DGS convention); undo transpose -> W2C
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)  # [1, 4, 4]

    # Colors
    sh_degree = None
    if override_color is not None:
        colors = override_color  # [N, 3]
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)  # [N, 3]
    else:
        colors = pc.get_features  
        sh_degree = pc.active_sh_degree
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        renders, alphas, info = rasterization(
            means=pc.get_xyz,
            quats=pc.get_rotation,          
            scales=pc.get_scaling * scaling_modifier,
            opacities=pc.get_opacity.squeeze(-1),  
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            backgrounds=bg_color.unsqueeze(0), 
            packed=False, 
        )
    rendered_image = renders[0].permute(2, 0, 1).float()  # [3, H, W]
    radii_2d = info["radii"][0]                   # [N, 2] (x/y screen radii)
    radii = radii_2d.max(dim=-1).values           # [N] scalar screen radius
    means2D = info["means2d"]
    
    if means2D.requires_grad:
        means2D.retain_grad()

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": (radii_2d > 0).all(dim=-1),  # [N]
        "radii": radii,
    }

def render_pose(
    viewpoint_camera, 
    pc: GaussianModel, 
    args, 
    bg_color: torch.Tensor,
    viewmatrix: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None
):
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    fx = W / (2 * tanfovx)
    fy = H / (2 * tanfovy)
    K = torch.tensor(
        [[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]],
        device="cuda", 
        dtype=torch.float32
    ).unsqueeze(0)

    # viewmatrix coming in is already row-major W2C (no transpose needed for gsplat)
    viewmat = viewmatrix.unsqueeze(0)  # [1, 4, 4]

    sh_degree = None
    if override_color is not None:
        colors = override_color
    elif args.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors = pc.get_features
        sh_degree = pc.active_sh_degree

    renders, alphas, info = rasterization(
        means=pc.get_xyz,
        quats=pc.get_rotation,
        scales=pc.get_scaling * scaling_modifier,
        opacities=pc.get_opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree,
        backgrounds=bg_color.unsqueeze(0),  
        packed=False,  
    )

    rendered_image = renders[0].permute(2, 0, 1) 
    radii_2d = info["radii"][0]             
    radii = radii_2d.max(dim=-1).values     

    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"][0], 
        "visibility_filter": (radii_2d > 0).all(dim=-1),  
        "radii": radii,
    }