import os
import sys
import cv2
import copy
import time
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import torch.optim 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from importlib import import_module 
import torchvision.transforms.functional as TF

args = argparse.ArgumentParser()
args.add_argument('--gpu', type=str, default='0')
args.add_argument('--random_seed', type=int, default=2025)
args.add_argument('--epochs', type=int, default=50)
args.add_argument('--exp_name', type=str, default='base') 
args.add_argument('--mode', type=str, default='VP', choices=['VP', 'FT'])
args.add_argument('--dataset', type=str, default='ibims', choices=['ibims', 'ddad'])
args.add_argument('--dataset_path', type=str, default='/workspace/data_all')

args = args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from unidepth_custom.models import UniDepthV2

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_scale_and_shift(predicted_depth, sparse_depth):

    valid_mask = (sparse_depth > 0)
    
    pred_valid = predicted_depth[valid_mask]   
    sparse_valid = sparse_depth[valid_mask]    
    
    if pred_valid.numel() == 0:
        device = predicted_depth.device
        dtype = predicted_depth.dtype
        return torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
    
    X = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
    
    a = torch.pinverse(X) @ sparse_valid 
    scale = a[0]
    shift = a[1]
    
    return scale, shift


def main(args):    
    print(f"Dataset: {args.dataset}, Mode: {args.mode}")
    tmp_dataset = import_module('dataloader.' + args.dataset)
    dataset = tmp_dataset.valset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    foundation_model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2old-vitl14").cuda() 
    
    rmse_mean_ft, mae_mean_ft = 0.0, 0.0
    rmse_mean_vp, mae_mean_vp = 0.0, 0.0
    
    for idx, sample in enumerate(dataloader):
        rgb = sample['rgb'].cuda()
        depth = sample['gt'].cuda()
        sparse_depth = sample['dep'].cuda()
        gt_mask = depth > 0
        sparse_mask = sparse_depth > 0

        
        if args.mode == 'VP':
            # Visual Prompt
            visual_prompt = torch.nn.Parameter(torch.zeros(1, 3, 476, 630, device='cuda'))
            optimizer = torch.optim.AdamW([{'params': visual_prompt, 'lr': 2e-3}])
            
            pbar = tqdm.tqdm(total=args.epochs)
            rgb_uni = F.interpolate(rgb, (476, 630), mode='bilinear', align_corners=False)
            
            for epoch in range(args.epochs):               
                new_rgb = rgb_uni + visual_prompt
                pre_depth_ = foundation_model({'image': new_rgb, 'depth': sparse_depth}, {})['depth']
                
                scale, shift = compute_scale_and_shift(pre_depth_, sparse_depth)    
                pre_depth = pre_depth_ * scale + shift    
                    
                loss_l1 = F.l1_loss(pre_depth[sparse_mask], sparse_depth[sparse_mask])
                loss_rmse = torch.sqrt(((pre_depth[sparse_mask] - sparse_depth[sparse_mask]) ** 2).mean())
                loss = loss_l1 + loss_rmse

                optimizer.zero_grad()
                loss.backward()          
                optimizer.step()

                pbar.set_description(f'exp: {args.exp_name} l1: {loss_l1.item():.4f} rmse: {loss_rmse.item():.4f}')
                pbar.update()
            with torch.no_grad():
                rmse_vp, mae_vp = torch.sqrt(((pre_depth[gt_mask] - depth[gt_mask]) ** 2).mean()), torch.abs(pre_depth[gt_mask] - depth[gt_mask]).mean()
                rmse_mean_vp += rmse_vp.item()
                mae_mean_vp += mae_vp.item()
                        
            pbar.close()
            
        
        elif args.mode == 'FT':
            # Fine-tuning
            # foundation_model_ft.load_state_dict(torch.load('/workspace/pretrained/unidepth_v2_weight.pth'))
            optimizer_ft = torch.optim.AdamW([{'params': foundation_model.parameters(), 'lr': 1e-6}])
            
            pbar = tqdm.tqdm(total=args.epochs)
            rgb_uni = F.interpolate(rgb, (476, 630), mode='bilinear', align_corners=False)
            
            for epoch in range(args.epochs):               
                pre_depth__ = foundation_model({'image': rgb_uni, 'depth': sparse_depth}, {})['depth'] 
                            
                scale, shift = compute_scale_and_shift(pre_depth__, sparse_depth)    
                pre_depth__ = pre_depth__ * scale + shift    
                    
                loss_l1_ = F.l1_loss(pre_depth__[sparse_mask], sparse_depth[sparse_mask])
                loss_rmse_ = torch.sqrt(((pre_depth__[sparse_mask] - sparse_depth[sparse_mask]) ** 2).mean())
                loss_ = loss_l1_ + loss_rmse_

                optimizer_ft.zero_grad()
                loss_.backward()          
                optimizer_ft.step()

                pbar.set_description(f'exp: {args.exp_name} l1: {loss_l1_.item():.4f} rmse: {loss_rmse_.item():.4f}')
                pbar.update()
                
            with torch.no_grad():
                rmse_ft, mae_ft = torch.sqrt(((pre_depth__[gt_mask] - depth[gt_mask]) ** 2).mean()), torch.abs(pre_depth__[gt_mask] - depth[gt_mask]).mean()
                rmse_mean_ft += rmse_ft.item()
                mae_mean_ft += mae_ft.item()
            
            pbar.close()
            
        if args.mode == 'VP':
            print(f'RMSE: {rmse_mean_vp}, MAE: {mae_mean_vp} idx: {idx}')
        elif args.mode == 'FT':
            print(f'RMSE: {rmse_mean_ft}, MAE: {mae_mean_ft} idx: {idx}')
        else:
            print("Invalid model")
            
if __name__ == "__main__":
    main(args)

