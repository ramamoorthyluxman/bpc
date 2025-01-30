// TODO(Yadunund): Copyright.

#ifndef IBPC_POSE_ESTIMATOR_HPP_
#define IBPC_POSE_ESTIMATOR_HPP_

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "ibpc_interfaces/msg/camera.hpp"
#include "ibpc_interfaces/msg/photoneo.hpp"
#include "ibpc_interfaces/msg/pose_estimate.hpp"
#include "ibpc_interfaces/srv/get_pose_estimates.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "image_geometry/pinhole_camera_model.hpp"
#include "rclcpp/rclcpp.hpp"


namespace ibpc
{
//==================================================================================================
class PoseEstimator : public rclcpp::Node
{
public:
  using Camera = ibpc_interfaces::msg::Camera;
  using GetPoseEstimates = ibpc_interfaces::srv::GetPoseEstimates;
  using Photoneo = ibpc_interfaces::msg::Photoneo;
  using PoseEstimate = ibpc_interfaces::msg::PoseEstimate;

  /// @brief Construct a PoseEstimator
  /// @param options Options for the ROS 2 node.
  explicit PoseEstimator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /// @brief Get the path to the directory in the local filesystem that contains various models
  ///   and other files required for inference.
  /// @return Path to model directory which can be used to load any models.
  const std::filesystem::path & model_dir() const;

  /// @brief Function for participants to implement to return pose estimates for objects in a scene.
  /// @param object_ids A list of object IDs in the scene whose pose should be estimated.
  /// @param rgb A cv::Mat containing the RGB image.
  /// @param rgb_camera_model The camera model of the RGB sensor.
  /// @param depth A cv::Mat containing the depth image.
  /// @param depth_camera_model The camera model of the depth sensor.
  /// @param polarized A cv::Mat containing the polarized image.
  /// @param polarized_camera_model The camera model of the polarized sensor.
  /// @return List of objects detected in the scene as a PoseEstimate message.
  /// @note For details on cv::Mat see https://docs.opencv.org/4.6.0/d3/d63/classcv_1_1Mat.html
  /// @note For details on image_geometry::PinholeCameraModel see
  std::vector<PoseEstimate> get_pose_estimates(
    const std::vector<uint64_t> & object_ids,
    const Camera & cam_1,
    const Camera & cam_2,
    const Camera & cam_3,
    const Photoneo & cam_4
  );

private:
  std::filesystem::path model_dir_;
  image_geometry::PinholeCameraModel rgb_camera_model_;
  image_geometry::PinholeCameraModel depth_camera_model_;
  image_geometry::PinholeCameraModel polarized_camera_model_;
  std::shared_ptr<rclcpp::Service<GetPoseEstimates>> srv_;
};

}  // namespace ibpc

#endif  // IBPC_POSE_ESTIMATOR_HPP_
