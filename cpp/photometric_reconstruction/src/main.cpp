#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/Polynomials>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>



// #define EIGEN_RUNTIME_NO_MALLOC

// Get Cx, Cy, fx, fy and distortion coefficients k1-4 for Kannala Brandt projection model
// TODO: tune with calibration
std::vector<double> get_camera_params() {
  return {735.37, 552.80, 717.21, 717.48, -0.13893, -1.2396e-3, 9.1258e-4, -4.0716e-5};
}


// Huber loss function
auto huber(auto value, auto huber_thresh) {
  if (value <= huber_thresh) return value;
  return 2.0*sqrt(value) - 1.0;
}


// Simple unprojection model
std::vector<double> unproject_pinhole(cv::Vec3b pixel) {
  std::vector<double> camera_params = get_camera_params();
  double cx = camera_params[0];
  double cy = camera_params[1];
  double fx = camera_params[2];
  double fy = camera_params[3];
  
  double mx = (pixel[0] - cx) / fx;
  double my = (pixel[1] - cy) / fy;
  double z = 1 / sqrt(mx*mx + my*my + 1);
  double x = mx * z;
  double y = my * z;
  return {x, y, z};
}


// TODO: Possibly replace with fisheye for simplification
std::vector<double> unproject_camera_model(cv::Vec3b pixel) {
  std::vector<double> camera_params = get_camera_params();
  double cx = camera_params[0];
  double cy = camera_params[1];
  double fx = camera_params[2];
  double fy = camera_params[3];
  double k1 = camera_params[4];
  double k2 = camera_params[5];
  double k3 = camera_params[6];
  double k4 = camera_params[7];

  // Unproject from 2D pixel to 3D point in the world
  double mx = (pixel[0] - cx) / fx;
  double my = (pixel[1] - cy) / fy;
  double r = sqrt(pow(mx, 2) + pow(my, 2));

  // Find roots for model k4(x^9) + k3(x^7) + k2(x^5) + k1(x^3) + x = r where x=theta
  Eigen::VectorXd coeffs(10);
  coeffs << -1*r, 1.0, 0.0, k1, 0.0, k2, 0.0, k3, 0.0, k4; 
  Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
  solver.compute(coeffs); // TODO: this is the bottleneck, is there a faster way to do this??
  const Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType &roots = solver.roots();
  for (auto root : roots) {
    // Check if complex part of root is 0, if so return solution
    if (root.imag() == 0) {
      return {mx, my, (double) root.real()};
    }
  }
  return {mx, my, 1};
}


// Get photometric model estimate
double calibrated_photometric_endoscope_model(double x, double y, double z, double k, double g_t, double gamma) {
  // Light spread function
  double mu_prime = abs(pow(cos(x), k));

  // BRDF
  double f_r_theta = 1/M_PI;

  // Distance to world coord from camera center
  double xc_to_pixel = sqrt(pow(x,2) + pow(y,2) + pow(z,2)); 

  // Get theta from angle of reflectance
  double theta = 2 * (acos(sqrt(pow(x,2) + pow(y,2)) / xc_to_pixel));

  // Solve for and intensity estimate for depth
  double L = (mu_prime / xc_to_pixel) * f_r_theta * cos(theta) * g_t;
  return pow(abs(L), gamma);
}


struct DepthEstimationNLS {

  // Member variables and constructor
  cv::Vec3b u;
  double intensity_gradient;
  DepthEstimationNLS(cv::Vec3b pixel_, double intensity_gradient_) : u(pixel_), intensity_gradient(intensity_gradient_) {}
  
  // Create instance
  static ceres::CostFunction* create(cv::Vec3b pixel, double intensity_gradient)
  {
    auto functor = new DepthEstimationNLS(pixel, intensity_gradient);
    return new ceres::AutoDiffCostFunction<DepthEstimationNLS, 1, 1>(functor);
  }

  // Operator for solving
  template <typename T>
  bool operator() (const T* const depth, T* residual) const {

    // Initial depth and its gradient
    T d = depth[0];

    // Tunable params
    double k = 2.;
    double g_t = 2.;
    double gamma = 2.;
    double huber_thresh = 1e-4;
    double regularization_lambda = 0.5;

    // std::vector<double> world_coord = unproject_camera_model(u); // angle to use in photometric model
    std::vector<double> world_coord = unproject_pinhole(u); // angle to use in photometric model
    double x = world_coord[0];
    double y = world_coord[1];
    double z = world_coord[2];

    // std::cout << "depth " << d << " world coords " << x << " " << y << " " << z << std::endl;

    // Light spread function
    double cos_alpha = z / sqrt(x*x + y*y + z*z); // off-axis angle from z
    double mu_prime = abs(pow(cos_alpha, k));

    // BRDF
    double f_r_theta = 1/M_PI;

    // Distance to world coord from camera center, should just be d
    // T xc_to_pixel = sqrt(x*x + y*y + d*d); 

    // Get theta from angle of reflectance
    // T theta = 2.0 * (acos(sqrt(pow(x,2) + pow(y,2)) / xc_to_pixel));
    double theta = 2.0 * acos(cos_alpha); // TODO: in radians?

    // Solve for and intensity estimate for depth
    T L = (mu_prime / (d*d)) * f_r_theta * cos(theta) * g_t;
    L = pow(abs(L), gamma);

    // std::cout << "light spread " << mu_prime << " brdf " << f_r_theta << " coord to cam center " << d << " ref angle " << theta << " intensity estimate " << L << std::endl;

    // Compute intensity of pixel
    auto b = u[0];
    auto g = u[1];
    auto r = u[2];
    double I = double ((r+g+b)/3);

    // Compute cost function
    T C = I - L;
    // std::cout << "I L C huberC " << I << " " << L << " " << C << " " << huber(C, huber_thresh) << std::endl;
    C = huber(C, huber_thresh);

    // Compute regularization function
    // T R;
    double intensity_gradient_adjusted = exp(-1 * intensity_gradient); // TODO: gradient used for smoothing here
    T depth_gradient = abs(d); // Computing actual gradient or hessian is difficult with ceres
    T R;
    R = intensity_gradient_adjusted * huber(depth_gradient, huber_thresh);
    std::cout << "R out " << R << " " << C + regularization_lambda * R << std::endl;
    residual[0] = C + regularization_lambda * R;

    return true;
  }
};


int main(int argc, char* argv[]) {
  // Update image
  std::string img_path = "/home/ishan/Downloads/frame-000055.color.jpg";
  cv::Mat cv_img = cv::imread(img_path, 1);
  if (!cv_img.data) {
      printf("No image data \n");
      return -1;
  }

  // Make grayscale output image of same size
  cv::Mat depth_map(cv_img.rows, cv_img.cols, CV_64FC1, (const double) 0);// Write image back out to depth map
  cv::Mat depth_map_init(depth_map.rows, depth_map.cols, CV_8UC1, (const cv::Scalar) 0);
  // #pragma omp parallel for
  // for (int row=0; row<cv_img.rows; row++) {
  //   for (int col=0; col<cv_img.cols; col++) {
  //     cv::Vec3b pixel = cv_img.at<cv::Vec3b>(row, col);
  //     auto b = pixel[0];
  //     auto g = pixel[1];
  //     auto r = pixel[2];
  //     double intensity = double((r+g+b)/3.0);
  //     double depth_init = 1.0 / sqrt(intensity);
  //     depth_map.at<double>(row, col) = depth_init;
  //     std::cout << row << " " << col << " " << cv_img.rows << " " << cv_img.cols << std::endl;
  //   }
  // }

  // cv::normalize(depth_map, depth_map_init, 255, 0, cv::NORM_MINMAX);
  // bool out = cv::imwrite("/home/ishan/Documents/research/bm2/reconstruction/photometric_approach/images/depth_map_init_cpp.png", depth_map_init);
  // if (!out) std::cout << "Image failed to write" << std::endl;
  // return 0;
  
  // Pass in initial depth estimate to solver params and create depth map
  std::vector<double> parameters(cv_img.cols, 0);

  cv::Mat laplacian = cv_img;
  cv::Laplacian(cv_img, laplacian, CV_64F);

  #pragma omp parallel for
  for (int row=0; row<cv_img.rows; row++) {
    for (int col=0; col<cv_img.cols; col++) {
      cv::Vec3b pixel = cv_img.at<cv::Vec3b>(row, col);
      auto b = pixel[0];
      auto g = pixel[1];
      auto r = pixel[2];
      double intensity = double((r+g+b)/3.0);
      double depth_init = 1.0 / sqrt(intensity); //TODO: is this negative??
      double depth = depth_init;

      ceres::Problem problem;
      auto img_grad_vec = laplacian.at<cv::Vec3b>(row, col);
      double img_grad = (img_grad_vec[0] + img_grad_vec[1] + img_grad_vec[2]) / 3.0;
      std::cout << "img grad "  << img_grad << std::endl;
      auto cost = DepthEstimationNLS::create(pixel, img_grad);
      problem.AddResidualBlock(cost, nullptr, &depth);

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = false;
      options.num_threads = omp_get_max_threads();
      ceres::Solver::Summary summary;
      Solve(options, &problem, &summary);

      std::cout << summary.BriefReport() << "\n";
      std::cout << "x : " << depth_init
                << " -> " << depth << "\n";
      depth_map.at<double>(row, col) = depth;
      
      // Check progress
      std::cout << row << " " << col << " " << cv_img.rows << " " << cv_img.cols << std::endl;
    }

  }

  // Write image back out to depth map
  cv::Mat depth_map_normalized(depth_map.rows, depth_map.cols, CV_8UC1, (const double) 0);
  cv::normalize(depth_map, depth_map_normalized, 255, 0, cv::NORM_MINMAX); 

  bool out = cv::imwrite("/home/ishan/Documents/research/bm2/reconstruction/photometric_approach/images/depth_map_cpp.png", depth_map_normalized);
  if (!out) std::cout << "Image failed to write" << std::endl;

  return 0;
}
