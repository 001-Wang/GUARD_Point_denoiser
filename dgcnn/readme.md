test dgcnn_sngp:
python -m dgcnn.test_all --method sngp   --pnfront_module pointnet2.models.sngp_s2_6layers   --pnfront_ckpt pointnet2/log/sngp/checkpoints/best_model.ckpt   --precision_path pointnet2/log/sngp/checkpoints/P_clean.pt  --root ..\pointnet2\data\shapenet_c_add\add_ghostcluster_s5 

test dgcnn_pointcvar:
python -m dgcnn.test_all_dgcnn_unified --method pointcvar  --root ..\pointnet2\data\shapenet_c_add\add_ghostcluster_s5  