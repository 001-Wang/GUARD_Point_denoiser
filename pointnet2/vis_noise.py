import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def show_interactive(xyz, label=None):
    """Open an interactive window (white background, color by label)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if label is not None:
        cmap = plt.get_cmap("tab20")  # 20 distinct colors
        colors = np.array([cmap(int(l) % 20)[:3] for l in label])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0, 0, 0])  # black points if no label

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Labels", width=960, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # white background
    opt.point_size = 3.0
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------
# Load your .txt file: x y z label
# ---------------------------------------------------------------------
path = r"data_prepare\shapenet_c_add\add_ghostcluster_s5\03642806\8d70fb6adc63e21eb7e0383b9609fa5.txt"

# load file and split xyz / labels
data = np.loadtxt(path).astype(np.float32)
xyz = data[:, :3]
labels = data[:, -1].astype(int)

# visualize
show_interactive(xyz, label=labels)
