import numpy as np

import torch
import torch.nn.functional as F

from jaxtyping import Float
from torch import nn, Tensor
import matplotlib.pyplot as plt

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def homogenize_matrices(
    matrices: Float[Tensor, "batch 3 4"],
) -> Float[Tensor, "batch 4 4"]:
    """Convert batched matrices (3 4) to (4 4)"""
    bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=matrices.dtype, device=matrices.device).repeat(matrices.shape[0], 1, 1)
    return torch.cat([matrices, bottom_row], dim=1)


def get_pose0(model, im):
    # im: B, 3, H, W
    imgs_pair = torch.cat([im, im], dim=1) # B, 6, H, W
    poses = model(imgs_pair)
    P = pose_vec2mat(poses)
    return P

def get_relative_pose(model, img1, img2):
    poses = model(torch.cat([img1, img2], dim=1))
    theta = pose_vec2mat(poses)[0]
    return theta

    img_src = img1.repeat(2, 1, 1, 1)
    img_tgt = torch.stack([img1.squeeze(0), img2.squeeze(0)], dim=0)
    imgs_pair = torch.cat([img_src, img_tgt], dim=1) # 2, 6, H, W
    poses = model(imgs_pair)
    Rt0 = pose_vec2mat(poses[0:1])
    Rt1 = pose_vec2mat(poses[1:2])
    R0 = Rt0[...,:3]
    R1 = Rt1[...,:3]
    t0 = Rt0[...,3]
    t1 = Rt1[...,3]
    R = R0.transpose(1,2) @ R1
    t = R0.transpose(1,2) @ (t1-t0).unsqueeze(2)
    P = torch.cat([R,t], dim=2)
    return P

def construct_trajectory(poses):
    # poses: t, 3, 4
    # return: t, 3, 4
    cur_t = torch.tensor([0,0,0]).to(poses[0].device)
    cur_R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).to(poses[0].device)

    cur_ts, cur_Rs = [], []
    for pose in poses:
        r = pose[:,:3]
        t = pose[:,3]
        cur_t = cur_t + cur_R @ t
        cur_R = r @ cur_R
        cur_ts.append(cur_t)
        cur_Rs.append(cur_R)

    R = torch.stack(cur_Rs)
    t = torch.stack(cur_ts)
    P = torch.cat([R,t.unsqueeze(2)], dim=2)
    return P

def pose_inverse_4x4(mat: torch.Tensor, use_inverse: bool=False) -> torch.Tensor:
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, V, 4 4) or (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 4:
        # must be (B, V, 4, 4)
        out_mat[:, :, 3, 3] = 1
        R,t = mat[:, :, :3, :3],mat[:,:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :, :3] = pose_inv
    elif len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R,t = mat[:, :3, :3],mat[:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R,t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat



def fig2img(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    ncols, nrows = fig.canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    img_array = img_array[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    fig.canvas.flush_events()

    return img_array[:, :, :3]