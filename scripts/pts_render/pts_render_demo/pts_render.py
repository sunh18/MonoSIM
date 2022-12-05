import os

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import scipy.io as sio
torch.manual_seed(42)
EPS = 1e-2


class Options:
    def __init__(self):
        self.rad_pow = 2.  # Exponent to raise the radius to when computing distance
                           # (default is euclidean, when rad_pow=2).
        self.tau = 1.  # gamma: the power to raise the distance to.
        self.accumulation = "wsum"   # choices=("wsum", "wsumnorm", "alphacomposite"),
        self.learn_default_feature = False
        self.radius = 4   # Radius of points to project
        self.image_size = (352, 1216)
        self.pp_pixel = 2  # the number of points to conisder in the z-buffer.


class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created (H, W)
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        C=64,
        learn_feature=True,
        radius=1.5,
        size=(256, 256),
        points_per_pixel=8,
        opts=None,
    ):
        super().__init__()
        if learn_feature:
            default_feature = nn.Parameter(torch.randn(1, C, 1))
            self.register_parameter("default_feature", default_feature)
        else:
            default_feature = torch.zeros(1, C, 1)
            self.register_buffer("default_feature", default_feature)

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        default_opts = Options()
        if opts is None:
            self.opts = default_opts
        else:
            self.opts = opts

    def forward(self, pts3D, src):
        bs = src.size(0)
        image_size = self.size

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)

        # Add on the default feature to the end of the src
        # src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

        radius = float(self.radius) / float(max(image_size)) * 2.0

        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        points_idx, _, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        )

        dist = dist / pow(radius, self.opts.rad_pow)

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.opts.tau)
            .permute(0, 3, 1, 2)
        )

        if self.opts.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

        return transformed_src_alphas


class PtsManipulator(nn.Module):
    def __init__(self, image_size, C=64, opts=None):
        super().__init__()
        default_opts = Options()
        if opts is None:
            self.opts = default_opts
        else:
            self.opts = opts
        self.image_size = image_size
        self.splatter = RasterizePointsXYsBlending(C=C, learn_feature=self.opts.learn_default_feature,
                                                   radius=self.opts.radius, size=image_size,
                                                   points_per_pixel=self.opts.pp_pixel, opts=self.opts)

    def project_pts(self, pts3D, K):
        """
        :param pts3D: Bx3xN, input points
        :param K: Bx3x3, camera intrinsics
        :return: Bx3xN, points in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
            e.g. for image size (H, W) = (64, 128)
           Height NDC range: [-1, 1]
           Width NDC range: [-2, 2]
        """
        # PERFORM PROJECTION
        # normalize K
        H, W = self.image_size
        ws = min(H, W)
        nK = torch.zeros_like(K)
        nK[:, 0, :] = K[:, 0, :] / ws
        nK[:, 1, :] = K[:, 1, :] / ws
        nK[:, 2, 2] = 1.

        # Add intrinsics
        xy_proj = nK.bmm(pts3D)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((2.0 * xy_proj[:, 0:1, :] / -zs + W/ws,
                             2.0 * xy_proj[:, 1:2, :] / -zs + H/ws,
                             xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10

        return sampler

    def forward(self, src_feat, pts3D, K):
        """
        :param src_feat: BxCxN, point features
        :param pts3D: Bx3xN, points
        :param K: Bx3x3, camera intrinsics
        :return: rendered feature map
        """
        # Now project these points into a new view

        pts3D = self.project_pts(pts3D, K)
        pointcloud = pts3D.permute(0, 2, 1).contiguous()  # B, N, 3
        result = self.splatter(pointcloud, src_feat)

        return result


if __name__ == '__main__':
    import numpy as np
    import cv2
    opts = Options()
    K = np.array([[721.5377, 0, 596.5593],
                  [0, 721.5377, 149.8540],
                  [0, 0, 1]], dtype=np.float32)
    K = torch.from_numpy(K).unsqueeze(0).cuda()
    pts = torch.from_numpy(np.load('/data/szb/M3D-RPN/scripts/pts_render/pts_render/test_data/points.npy').astype(np.float32)).unsqueeze(0).cuda()
   
    # pts = np.load('/data/szb/M3D-RPN/scripts/pts_render/pts_render/test_data/points.npy').astype(np.float32)
    # pts_name = '/data/szb/M3D-RPN/output/test_pts.mat'
    # sio.savemat(pts_name, {'pts': pts})  
    
    rgb = torch.from_numpy(np.load('/data/szb/M3D-RPN/scripts/pts_render/pts_render/test_data/rgb.npy').astype(np.float32)).unsqueeze(0).cuda()
    
    pts_manipulator = PtsManipulator((352, 1216), 3, opts)
    res = pts_manipulator(rgb, pts, K).detach().squeeze().permute(1,2,0).data.cpu().numpy()*255
    cv2.imshow('render', res.astype('uint8'))
    cv2.waitKey(0)