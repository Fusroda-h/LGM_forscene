import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer
from core.utils import get_rays

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
            
        # dataset param from pixelsplat
        # self.encoder = encoder
        # self.data_shim = get_data_shim(self.encoder)
        


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images):
        # images: [B, 2, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*2, 14, h, w]
        x = self.conv(x) # [B*2, 14, h, w]

        x = x.reshape(B, V, 14, self.opt.splat_size, self.opt.splat_size)
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians


    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0
        
        # batch: BatchedExample = self.data_shim(batch) # B 4 3 h w
        _b, _v, _, _h, _w = data["target"]["image"].shape
        
    
        # images = data['input'] # [B, 4, 9, h, W], input features
        images = data["context"]['image'] # b 2 3 h w
        
        _intrin = data["target"]["intrinsics"] # b v 3 3
        cam_poses = data["target"]["extrinsics"]# b v 4 4
        
        g2c = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32,device=images.device)
        c2g = torch.inverse(g2c)
        
        # _fovy = np.rad2deg(2 * np.arctan(h/2*_intrin[0,0,1,1].item()))

        # TODO: you may have a different camera system
        for i,cam_views in enumerate(cam_poses):
            for j,_c2w in enumerate(cam_views):
                # w2c to c2w
                c2w = _c2w.clone()
                
                c2w = g2c @ c2w @ c2g

                # blender world + opencv cam --> opengl world & cam
                #c2w[1] *= -1
                #c2w[[1, 2]] = c2w[[2, 1]]
                #c2w[:3, 1:3] *= -1 # invert up and forward direction

                # scale up radius to fully use the [-1, 1]^3 space!
                #c2w[:3, 3] *= self.opt.cam_radius / 1.5  # 1.5 is the default scale
                # c2w[:3, 3] /= torch.norm(c2w[:3,3])*1.5
                cam_poses[i,j] = c2w

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32,device=images.device) @ torch.inverse(cam_poses[0])
        # Normalized
        for i,c in enumerate(cam_poses):
            transform = torch.inverse(cam_poses[0])
            #transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32,device=images.device) @ torch.inverse(cam_poses[0])
            # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1/torch.norm(c[0,:3,3])], [0, 0, 0, 1]], dtype=torch.float32,device=images.device) @ torch.inverse(c[0])
            cam_poses[i] = transform.unsqueeze(0) @ c  # [V, 4, 4]

        
        # build rays for input views
        ray_embedded_images = torch.empty((_b, 2, 9, _h, _w),dtype=torch.float32, device=images.device)
        for i in range(_b):
            rays_embeddings = []
            for j in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses[i,j], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)
            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            # print(torch.cat([images[i], rays_embeddings], dim=1).shape)
            ray_embedded_images[i] =  torch.cat([images[i], rays_embeddings], dim=1) # [V=2, 9, H, W]
                        
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(ray_embedded_images) # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # default camera intrinsics
        tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        proj_mat = torch.zeros((4, 4), dtype=torch.float32, device=gaussians.device)
        proj_mat[0, 0] = 1 / tan_half_fov#_fovy
        proj_mat[1, 1] = 1 / tan_half_fov#_fovy
        proj_mat[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        proj_mat[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        proj_mat[2, 3] = 1

        proj_matrices = einops.repeat(proj_mat,'n m -> b v n m', b=_b,v=_v)
        
        # opengl to colmap camera for gaussian renderer
        # cam_poses[:, :, :3, 1:3] *= -1 # invert up & forward direction
        for i,_c2w in enumerate(cam_poses):
            for j, __c2w in enumerate(_c2w):
                cam_poses[i,j] = c2g @ __c2w @ g2c

        # colmap cameras needed by gaussian rasterizer 
        cam_view = torch.inverse(cam_poses).transpose(2, 3) 
        cam_view_proj = einops.einsum(cam_view, proj_matrices,'b v i k, b v k j -> b v i j')
        cam_pos = - cam_poses[:, :, :3, 3]
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, cam_view, cam_view_proj, cam_pos, bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data["target"]["image"] # [B, V, 3, output_size, output_size], ground-truth novel views
        # gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        # gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        loss_mse = F.mse_loss(pred_images, gt_images) # + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
    
    # def forward(self, data, step_ratio=1):
    #     # data: output of the dataloader
    #     # return: loss

    #     results = {}
    #     loss = 0

    #     images = data['input'] # [B, 4, 9, h, W], input features
        
    #     # use the first view to predict gaussians
    #     gaussians = self.forward_gaussians(images) # [B, N, 14]

    #     results['gaussians'] = gaussians

    #     # always use white bg
    #     bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
    #     # use the other views for rendering and supervision
    #     results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
    #     pred_images = results['image'] # [B, V, C, output_size, output_size]
    #     pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

    #     results['images_pred'] = pred_images
    #     results['alphas_pred'] = pred_alphas

    #     gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
    #     gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

    #     gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

    #     loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
    #     loss = loss + loss_mse

    #     if self.opt.lambda_lpips > 0:
    #         loss_lpips = self.lpips_loss(
    #             # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
    #             # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
    #             # downsampled to at most 256 to reduce memory cost
    #             F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
    #             F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
    #         ).mean()
    #         results['loss_lpips'] = loss_lpips
    #         loss = loss + self.opt.lambda_lpips * loss_lpips
            
    #     results['loss'] = loss

    #     # metric
    #     with torch.no_grad():
    #         psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
    #         results['psnr'] = psnr

    #     return results
