import tyro
import time
import random
from pathlib import Path

import torch
from core.options import AllConfigs
from core.models import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui

from pixelsrc.dataset.data_module import DataModule
from pixelsrc.misc.step_tracker import StepTracker
from pixelsrc.dataset import DatasetRE10kCfg, data_module, view_sampler

def main():    
    opt = tyro.cli(AllConfigs)
    # Config MVsplat
    # cfg = load_typed_root_config(cfg_dict)
    # set_cfg(cfg_dict)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dispatch_batches=False
        # kwargs_handlers=[ddp_kwargs],
    )

    # model
    model = LGM(opt)

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    # data
    # if opt.data_mode == 's3':
    #     from core.provider_objaverse import ObjaverseDataset as Dataset
    # else:
    #     raise NotImplementedError

    # train_dataset = Dataset(opt, training=True)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    step_tracker = StepTracker()
    
    cfg_dataset = DatasetRE10kCfg(image_shape=[256, 256], background_color=[0.0, 0.0, 0.0], cameras_are_circular=False,
                                  overfit_to_scene=None, view_sampler=view_sampler.ViewSamplerBoundedCfg(name='bounded', num_context_views=2, num_target_views=4, min_distance_between_context_views=45, 
                                                       max_distance_between_context_views=45, min_distance_to_context_views=0, warm_up_steps=150000, 
                                                       initial_min_distance_between_context_views=25, initial_max_distance_between_context_views=25), name='re10k', 
                    roots=[Path('/mnt1/dataset/pixelsplat/re10k')], baseline_epsilon=0.001, max_fov=100.0, make_baseline_1=True, augment=True) 
    cfg_dataloader = data_module.DataLoaderCfg(train=data_module.DataLoaderStageCfg(batch_size=7, num_workers=8, persistent_workers=True, seed=1234), 
                  test=data_module.DataLoaderStageCfg(batch_size=1, num_workers=4, persistent_workers=False, seed=2345), 
                  val=data_module.DataLoaderStageCfg(batch_size=1, num_workers=1, persistent_workers=True, seed=3456))
    

    # cfg_dataset = DatasetRE10kCfg(image_shape=[256, 256], background_color=[0.0, 0.0, 0.0], cameras_are_circular=False, overfit_to_scene=None, 
    #                 view_sampler=view_sampler.ViewSamplerEvaluationCfg(name='train',
    #                 index_path=Path('pixelsrc/assets/evaluation_index_re10k.json'), num_context_views=2), name='re10k',
    #                 roots=[Path('datasets/dataset/pixelsplat/re10k')], baseline_epsilon=0.001, max_fov=100.0, make_baseline_1=True, augment=True)
    
    # cfg_dataloader = data_module.DataLoaderCfg(train=data_module.DataLoaderStageCfg(batch_size=7, num_workers=16, persistent_workers=True, seed=1234), 
    #               test=data_module.DataLoaderStageCfg(batch_size=1, num_workers=4, persistent_workers=False, seed=2345),
    #               val=data_module.DataLoaderStageCfg(batch_size=1, num_workers=1, persistent_workers=True, seed=3456))
    
    
    train_dataloader = DataModule(
        cfg_dataset,
        cfg_dataloader,
        step_tracker,
    ).train_dataloader()
    
    test_dataloader = DataModule(
        cfg_dataset,
        cfg_dataloader,
        step_tracker,
    ).test_dataloader()

    # test_dataset = Dataset(opt, training=False)
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                # save log images
                if i % 500 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, opt.workspace)

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                
                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")



if __name__ == "__main__":
    main()
