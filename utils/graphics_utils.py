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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, cx, cy):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = top
    right = tanHalfFovX * znear
    left = right
    z_sign = 1.0

    # P = torch.zeros(4, 4)
    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)

    # P = torch.zeros([4, 4])
    # P[0, 0] = -1.0 / tanHalfFovX
    # P[1, 1] = -1.0 / tanHalfFovY
    # P[2, 2] = 1.0
    # P[3, 2] = 1.0

    W, H = 512, 512
    fx = fov2focal(fovX, W)
    fy = fov2focal(fovY, H)

    P = torch.zeros(4, 4, dtype=torch.float32)

    # X-axis mapping: (2 * x_pixel / W) - 1 => (2 * (fx * X_cam / Z_cam + cx) / W) - 1
    P[0, 0] = 2 * fx / W
    P[0, 2] = 2 * cx / W - 1  # Note: 1 - 2*cx/W to map pixel 0 to -1 and W-1 to 1

    # Y-axis mapping: 1 - (2 * y_pixel / H) => 1 - (2 * (fy * Y_cam / Z_cam + cy) / H)
    # This maps top (y=0) to 1, bottom (y=H-1) to -1.
    P[1, 1] = -2 * fy / H
    P[1, 2] = 1 - 2 * cy / H  # Note: 2*cy/H - 1 to map pixel 0 to 1 and H-1 to -1

    # Z-axis mapping: map [near, far] to [0, 1]
    # Z_ndc = (Z_cam - near) / (far - near)  (for linear depth)
    # For perspective Z: A + B/Z
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -zfar * znear / (zfar - znear)  # Or -far * near / (far - near) if camera looks down negative Z

    # Homogeneous coordinate W (perspective divide)
    P[3, 2] = 1.0  # This ensures the division by Z_camera (after scaling)

    return P


def getProjectionMatrix_(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))