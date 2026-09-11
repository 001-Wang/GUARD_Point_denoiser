<p align="center">
  <img src="assets/GUARD_logo.png" alt="GUARD Logo" width="420">
</p>

# GUARD
Early-stage **Geometric Uncertainty-Aware Robust Denoiser (GUARD)** for point cloud corruption removal and robust segmentation.

## Project page

Visit the public project page: **https://001-wang.github.io/GUARD_Point_denoiser/**

The same page is also mirrored at: https://guard-point-cloud-zuoxu.zuoxu.chatgpt.site/

This repository provides code, datasets, pretrained checkpoints, and evaluation pipelines for our GUARD framework based on early-stage geometric gaussian process.

---

## Reproducibility Environment

- **OS:** Ubuntu 20.04 / Windows 11  
- **Python:** 3.11  
- **CUDA:** 11.8  
- **GPU:** RTX 4080 / A100  
- **PyTorch:** 2.7.1  
- **cuDNN:** 12.6  

## Create Conda Environment
```bash
conda env create -f environment.yml
conda activate myenv
```



## Data Preparation

For Shapenetnet data, its an public dataset, you can get from the offcial website. after getting the orignal data, put them in the folder ./data_prepare/shapenet

### Ghost Corruption Data

then run 
```bash
  python make_pcc_ghostcluster.py 
    --json test_list.json 
    --dataset-root ./data_prepare/shapenet 
    --output-root data/shapenet_c_add/add_ghostcluster_s3 
    --severity 3 
    --clusters 1 --ratio 0.25 --rot-deg 10 --trans-range 0.05 --jitter-std 0.0
```
to get ghost corruption data

### PointNet-C Random Noise Corruption Data

run:
```bash
  python make_pcc_noisy_simple.py --list .\data_prepare\shapenet\train_test_split\shuffled_test_file_list.json
      --dataset-root ./data_prepare/shapenet 
      --output-root "data/shapenet_c_add"
```
to get pointnet-c random noise corruption data


### HDD Real Scan Data

for our HDD real Scan data, we will upload it after the paper publlished

but even you dont downdoad any data, we still prepare some pieces of data for you to visualization in the zip. you can run the command on the visualizaition part directly






## Evaluation

### ShapeNet Data



#### Baseline

pointnet2:
```bash
python -m pointnet2.test_partseg_baseline  --log_dir without_normal --root data_prepare\shapenet_c_add\shapenet     #(clean data)  
python -m pointnet2.test_partseg_baseline  --log_dir without_normal --root data_prepare\shapenet_c_add\add_global_s5  
```


dgcnn:
```bash
python -m dgcnn.test_partseg_baseline  --root data_prepare\shapenet_c_add\shapenet    #(clean data) 
python -m dgcnn.test_partseg_baseline  --root data_prepare\shapenet_c_add\add_ghostcluster_s5
```


#### PointNet2

test pn2_sngp:
```bash
python -m pointnet2.test_all_v6 --method sngp --time_include_post --root data_prepare\shapenet_c_add\add_ghostcluster_s5  
```

test pn2_pointcvar:
```bash
python -m pointnet2.test_all_v6 --method pointcvar --time_include_post --root data_prepare\shapenet_c_add\add_ghostcluster_s5 
```

#### DGCNN

test dgcnn_sngp:
```bash
python -m dgcnn.test_all_dgcnn_unified --method sngp   --pnfront_module pointnet2.models.sngp_s2_6layers   --pnfront_ckpt pointnet2/log/sngp/checkpoints/best_model.ckpt   --precision_path pointnet2/log/sngp/checkpoints/P_clean.pt  --root data_prepare\shapenet_c_add\add_ghostcluster_s5  
```

test dgcnn_pointcvar:
```bash
python -m dgcnn.test_all_dgcnn_unified --method pointcvar  --root data_prepare\shapenet_c_add\add_ghostcluster_s5  
```



### HDD Data

#### Baseline

```bash
python -m pointnet2.test_partseg_baseline_hdd  --log_dir without_normal_hdd --root data_prepare\hdd_data
python -m dgcnn.test_partseg_baseline_hdd --root data_prepare\hdd_data 
```

#### PointNet2

test pn2_sngp:
```bash
python -m pointnet2.test_all_v6_hdd --method sngp --time_include_post --root data_prepare\hdd_data --hdd_simple --keep 2048 
```

test pn2_pointcvar:
```bash
python -m pointnet2.test_all_v6_hdd --method pointcvar --time_include_post --root data_prepare\hdd_data --hdd_simple --keep 2048
```

#### DGCNN

test dgcnn_sngp:
```bash
python -m dgcnn.test_all_dgcnn_hdd --method sngp --root data_prepare\hdd_data  
```

test dgcnn_pointcvar:
```bash
python -m dgcnn.test_all_dgcnn_hdd --method pointcvar --root data_prepare\hdd_data
```

## Visualization

### ShapeNet

```bash
python -m pointnet2.visualization --file "data_prepare\shapenet_c_add\add_ghostcluster_s5\03636649\5c5119a226e1ce9934804d261199e1bf.txt"  

python -m pointnet2.visualization_pointcvar --file "data_prepare\shapenet_c_add\add_ghostcluster_s5\03467517\5fc56e6d220d775e381b7fbf79296afb.txt"  
```

### HDD

```bash
python -m pointnet2.viz_HDD --file data_prepare\hdd_data\01234567\29_0.txt
python -m pointnet2.viz_HDD_pointcvar --file data_prepare\hdd_data\01234567\29_0.txt 
```

## Training

```bash
train_chunkknn_partseg.py --data_root data_prepare\hdd_data --model sngp_hdd --num_part 5
train_chunkknn_partseg.py --data_root data_prepare\shapenet --model sngp_s2_6layers --num_part 50
```


## Acknowledgments

Our implementation builds upon several excellent open-source repositories.  
We sincerely thank the authors for their contributions to the community:

- **PointNet++ PyTorch version**  
  https://github.com/yanx27/Pointnet_Pointnet2_pytorch

- **DGCNN PyTorch version**  
  https://github.com/WangYueFt/dgcnn/tree/master/pytorch
