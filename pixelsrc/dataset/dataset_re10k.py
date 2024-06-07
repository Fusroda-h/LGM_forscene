import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

from core.options import Options


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        opt: Options,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.opt = opt

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                # if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                #     continue

                # Load the images.
                try:
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]
                    context_images = self.convert_images(context_images)
                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]
                    target_images = self.convert_images(target_images)
                except IndexError:
                    continue

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                if context_image_invalid or target_image_invalid:
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / scale,
                        "far": self.get_bound("far", len(context_indices)) / scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / scale,
                        "far": self.get_bound("far", len(target_indices)) / scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    # def objaverse_preprocessing(self,exp):
    #     # load num_views images
    #     images = []
    #     masks = []
    #     cam_poses = []
        
    #     vid_cnt = 0
        
    #     for image, c2w in zip(exp['context']['image'],exp['context']['extrinsics']):
            
    #         # TODO: you may have a different camera system
    #         # blender world + opencv cam --> opengl world & cam
    #         c2w[1] *= -1
    #         c2w[[1, 2]] = c2w[[2, 1]]
    #         c2w[:3, 1:3] *= -1 # invert up and forward direction

    #         # scale up radius to fully use the [-1, 1]^3 space!
    #         c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
          
    #         image = image.permute(2, 0, 1) # [4, 512, 512]
    #         images.append(image)
    #         cam_poses.append(c2w)

    #         vid_cnt += 1
    #         if vid_cnt == self.opt.num_views:
    #             break

    #     if vid_cnt < self.opt.num_views:
    #         print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
    #         n = self.opt.num_views - vid_cnt
    #         images = images + [images[-1]] * n
    #         masks = masks + [masks[-1]] * n
    #         cam_poses = cam_poses + [cam_poses[-1]] * n
          
    #     images = torch.stack(images, dim=0) # [V, C, H, W]
    #     masks = torch.stack(masks, dim=0) # [V, H, W]
    #     cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

    #     # normalized camera feats as in paper (transform the first pose to a fixed position)
    #     transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
    #     cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

    #     images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
    #     cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

    #     # data augmentation
    #     if self.training:
    #         # apply random grid distortion to simulate 3D inconsistency
    #         if random.random() < self.opt.prob_grid_distortion:
    #             images_input[1:] = grid_distortion(images_input[1:])
    #         # apply camera jittering (only to input!)
    #         if random.random() < self.opt.prob_cam_jitter:
    #             cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

    #     images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    #     # resize render ground-truth images, range still in [0, 1]
    #     results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
    #     results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

    #     # build rays for input views
    #     rays_embeddings = []
    #     for i in range(self.opt.num_input_views):
    #         rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
    #         rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
    #         rays_embeddings.append(rays_plucker)

     
    #     rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
    #     final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
    #     results['input'] = final_input

    #     # opengl to colmap camera for gaussian renderer
    #     cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
    #     # cameras needed by gaussian rasterizer
    #     cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    #     cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
    #     cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
    #     results['cam_view'] = cam_view
    #     results['cam_view_proj'] = cam_view_proj
    #     results['cam_pos'] = cam_pos

    #     return exp
    
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())
