import numpy as np


def get_intensity(pixel):
  r,g,b = pixel
  return (float(r) + float(g) + float(b)) / (255*3)


def get_camera_params():
  # returns cx, cy, fx, fy and distortion coefficients k1-4 for Kannala Brandt projection model
  return 735.37, 552.80, 717.21, 717.48, -0.13893, -1.2396e-3, 9.1258e-4, -4.0716e-5

def unproject_camera_model(u, d):
  # Kannala Brandt unprojection
  # map from 2D pixel value u to 3D point in world
  cx, cy, fx, fy, k1, k2, k3, k4 = get_camera_params()
  mx = (u[0] - cx) / fx
  my = (u[1] - cy) / fy
  r = (mx**2 + my**2) ** 0.5
  
  # Find roots for model k4(x^9) + k3(x^7) + k2(x^5) + k1(x^3) + x = r where x=theta
  coeffs = np.array([k4, 0, k3, 0, k2, 0, k1, 0, 1, -r])
  roots = np.roots(coeffs)
  theta = np.real(roots[0])
  
  # Get X,Y,Z output
  X = np.sin(theta) * (mx/r)
  Y = np.sin(theta) * (my/r)
  Z = np.cos(theta)
  return mx, my, theta