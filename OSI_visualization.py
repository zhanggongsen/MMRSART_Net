import numpy as np
import SimpleITK as sitk
import os
import open3d as o3d

PC_root = r"H:"

PC_path = os.path.join(PC_root, "PC.txt")
array_1 = np.loadtxt(PC_path)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(array_1)
# pcd.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([pcd])