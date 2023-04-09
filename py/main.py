from calibration import get_calibration
from utils import get_intensity, unproject_camera_model
from photometric_model import calibrated_photometric_endoscope_model, cost_function, regularization_function

from tqdm import tqdm
import numpy as np
import cv2
import tracemalloc
from multiprocessing import current_process, cpu_count, Process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# Globals for parallel processing
# global k, g_t, gamma, regularization_lambda, alpha, img, depth_map, gradient, energy_function


def compute_img_depths(img, iters=1000):

  img = 1/255 * img
  k, g_t, gamma = get_calibration(img)
  energy_function = np.zeros((img.shape[0], img.shape[1]))
  gradient = np.zeros((img.shape[0], img.shape[1]))
  # depth_map = np.sqrt(np.sum(img, axis=2))
  depth_map = np.zeros((img.shape[0], img.shape[1]))
  errors = []

  # Minimize energy function
  regularization_lambda = 0.5 # adjust
  alpha = 5 # adjust learning rate
  prev_energy_function = np.zeros((img.shape[0], img.shape[1]))
  prev_depth_map = np.zeros((img.shape[0], img.shape[1]))
  for i in tqdm(range(iters)):
    # Compute energy function for every pixel separately (as opposed to minimizing global energy function)
    for row in tqdm(range(img.shape[0])):
      for col in range(img.shape[1]):
        d = depth_map[row, col]
        u = img[row, col] # pixel
        x, y, z = unproject_camera_model(u, d) # angle to use in photometric model
        L = calibrated_photometric_endoscope_model(x, y, d, k, g_t, gamma) # TODO: Why is z never used??
        I = get_intensity(u)
        C = cost_function(I, L)
        R = regularization_function(gradient[row, col])
        energy_function[row, col] = C + regularization_lambda * R
        if row==200 and col==200:
          print(C, R, x, y, d, u, z)

        # Perform gradient descent for every pixel
        if i==0:
          depth_map[row, col] += .1
        else:
          gradient[row, col] = energy_function[row, col] - prev_energy_function[row, col]
          depth_dir = np.sign(depth_map[row, col] - prev_depth_map[row, col])
          depth_map[row, col] -= depth_dir * alpha * gradient[row, col]
        prev_energy_function[row, col] = energy_function[row, col]
        prev_depth_map[row, col] = depth_map[row, col]

    # TODO: Save energy function, depth map, and gradient
    error = np.sum(energy_function)
    print(f"Error: {error}")
    errors.append(error)
    with open("./errors.txt", "w") as f:
      for error in errors:
        f.write(str(error) + "\n")

    # Paper used trust region, this implementation just uses line search (gradient descent)
    print(gradient[200][200])
    # hessian = compute_hessian(energy_function)
    # trust_region_subproblem(energy_function, gradient, hessian)
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite(f'../images/depthmap_{i}.png', depth_map_img)

  return depth_map

# first take a random step, see if energy function goes down, update depths in opposite direction


# def compute_energy_function(row):
#   # print(depth_map[200][200])
#   for col in range(img.shape[1]):
#     print(col, img.shape[1], 3)
#     d = depth_map[row, col]

#     print(col, img.shape[1], 4)
#     u = img[row, col] # pixel
#     mx, my, x = unproject_camera_model(u, d) # angle to use in photometric model
#     print(f"dux {d} {u}, {x}")
#     L = calibrated_photometric_endoscope_model(mx, my, x, k, g_t, gamma)
#     I = get_intensity(u)
#     C = cost_function(I, L)
#     R = regularization_function(gradient[row, col])
#     energy_function[row, col] = C + regularization_lambda * R


# def compute_img_depths_parallel(img, iters=1000):
  
#   # Get initial depth estimate, energy function, and energy function gradient
#   energy_function = np.zeros((img.shape[0], img.shape[1]))
#   gradient = np.zeros((img.shape[0], img.shape[1]))
#   depth_map = np.sqrt(np.sum(img, axis=2))
#   print(depth_map[200][200])

#   # Minimize energy function
#   k, g_t, gamma = get_calibration(img)
#   regularization_lambda = 0.5 # adjust
#   alpha = 0.1 # adjust
#   prev_energy_function = None


#   for i in tqdm(range(iters)):
#     # Compute energy function for every pixel separately (as opposed to minimizing global energy function)
#     if i!=0: # Set to previous energy function
#       prev_energy_function = energy_function

#     ### Parallel section
#     num_procs = cpu_count() - 2
#     # row_cols = [[row, col] for row in range(img.shape[0]) for col in range(img.shape[1])] # Process each pixel
#     # pbar = tqdm(total=img.shape[0])
#     with ProcessPoolExecutor(num_procs) as exe:
#       row=0
#       while row < img.shape[0]:
#         fs = [
#           exe.submit(compute_energy_function, row+j)
#           for j in range(num_procs)
#           if row+j<img.shape[0]
#         ]
#         for _ in as_completed(fs): 
#           # pbar.update(1)
#           # row += 1
#           pass
#         row += num_procs
#     # pbar.close()

#     ###

#     # energy_function = compute_energy_function(img.shape[0], img.shape[1]) # parallelize this line

#     # Paper used trust region, this implementation just uses line search (gradient descent)
#     if i==0: # first iteration and every 10th iteration, take random step
#       step = np.random.random(depth_map.shape)
#       depth_map += step
#     else:
#       gradient = energy_function - prev_energy_function
#       # print(gradient)
#       depth_map -= alpha * gradient
#       # hessian = compute_hessian(energy_function)
#       # trust_region_subproblem(energy_function, gradient, hessian)
#     depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     cv2.imwrite(f'../images/depthmap_{i}.png', depth_map_img)

#   return depth_map





if __name__=='__main__':
  img = cv2.imread("../images/colon2.jpg")
  print(f"CPU Count: {cpu_count()}")
  print(compute_img_depths(img))