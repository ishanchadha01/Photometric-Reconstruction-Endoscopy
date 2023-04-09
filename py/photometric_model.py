import numpy as np


def calibrated_photometric_endoscope_model(x, y, z, k, g_t, gamma):
  mu_prime = light_spread_function(z, k)
  f_r_theta = 1/np.pi # Lambertian BRDF, might work better with value closer to 0 like 1/2pi
  xc_to_pixel = np.linalg.norm(np.array([x, y, z])) # Find distance from center of image to pixel
  theta = 2 * (np.arccos(np.linalg.norm(np.array([x,y])) / xc_to_pixel)) # Compute angle of incidence, and then find angle theta
  L = (mu_prime / xc_to_pixel) * f_r_theta * np.cos(theta) * g_t
  L  = np.power(np.abs(L), gamma) # TODO: might need to preserve sign
  return L


def cost_function(I, L, thresh=1e-4):
  # Huber norm of pixel intensity and photometric model
  if np.linalg.norm(I-L) <= thresh:
    norm = np.square(I-L) / (2*thresh)
  else:
    norm = np.abs(I-L) + (thresh/2)
  return norm
  

def regularization_function(grad, thresh=1e-4):
  #TODO: why is this always 0???
  g = np.exp(-np.linalg.norm(grad)) # Can adjust mutliplier of gradient (alpha) order or norm (beta)
  # Compute Huber norm of gradient
  if np.linalg.norm(grad) <= thresh:
    norm = np.power(np.linalg.norm(grad), 2) / (2*thresh)
  else:
    norm = np.abs(grad) + (thresh/2)
  return g * norm


def light_spread_function(x, k):
  # TODO: can try changing radians to degrees
  return np.power(np.abs(np.cos(x)), k) # TODO: might need to preserve sign