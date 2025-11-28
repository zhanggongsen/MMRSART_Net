import SimpleITK
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import SimpleITK as sitk

img=sitk.GetArrayFromImage(sitk.ReadImage(r""))
img1=img[0,:,:]
img2=img[1,:,:]
img1=(img1-np.min(img1))/(np.max(img1)-np.min(img1))
img2=(img2-np.min(img2))/(np.max(img2)-np.min(img2))
print(img1.shape)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# x=0
y1 = np.linspace(-1024/2, 1024/2, 1024)
z1 = np.linspace(-768/2, 768/2, 768)
y1, z1 = np.meshgrid(y1, z1)
x1 = np.zeros_like(y1)

ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, facecolors=plt.cm.gray(img1), shade=False)

# y=0
x2 = np.linspace(-1024/2, 1024/2, 1024)
z2 = np.linspace(-768/2, 768/2, 768)

x2, z2 = np.meshgrid(x2, z2)
y2 = np.zeros_like(x2)

ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, facecolors=plt.cm.gray(img2), shade=False)
ax.axis('off')
ax.view_init(elev=30, azim=45)
plt.show()