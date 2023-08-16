import torch
import tifffile as tiff
import glob
import os
import numpy as np

root = '/home/ziyi/Projects/Diffusion_Inference/out/cond_model/ddim/monte_carlo/step400_eta10_thres20_blur3/results/0'# '/home/ziyi/Projects/Diffusion_Inference/out/cond_model/ddim/monte_carlo/st20_tiny_step400_eta10_thres20/results/0' # /home/ziyi/Projects/Diffusion_Inference/out/cond_model/ddim/monte_carlo/step400_eta05/results/0
ids_plus = os.listdir(root + '/GT')
ids = set([i.rsplit('_', 1)[0] for i in ids_plus])

os.makedirs(root + 'Compose_res/', exist_ok=True)
for id in ids:
    tmp = []
    print(id)
    for num in range(20):
        cur_gt_path = root + '/GT/' + id + f'_{str(num).zfill(3)}.tif'
        pain_img = np.expand_dims(tiff.imread(cur_gt_path), 0).astype(np.float32)
        out_img = np.expand_dims(tiff.imread(cur_gt_path.replace('GT', 'Out')), 0).astype(np.float32)

        res = (pain_img - out_img)
        # print(res.max(), res.min())
        res[res<0] = 0
        # print(res.max(), res.min())
        tmp.append(res)

    tmp = np.concatenate(tmp, 0)
    mean_tmp = np.mean(tmp, 0)
    std_tmp = np.std(tmp, 0)
    var_tmp = np.var(tmp, 0)
    os.makedirs(root + 'Compose_var/', exist_ok=True)
    os.makedirs(root + 'Compose_mean/', exist_ok=True)
    heat_map = mean_tmp / (std_tmp + 0.0001)
    # print(id, '     ', heat_map.max(), heat_map.min())
    # heat_map = (((heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())) * 255).astype(np.uint8)
    tiff.imwrite(root + 'Compose_mean/' + id + '.tif', mean_tmp)
    tiff.imwrite(root + 'Compose_res/' + id +'.tif', heat_map)
    tiff.imwrite(root + 'Compose_var/' + id + '.tif', var_tmp)