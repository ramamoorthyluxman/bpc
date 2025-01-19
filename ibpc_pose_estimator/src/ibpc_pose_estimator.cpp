// TODO(Yadunund): Copyright.

#include "ibpc_pose_estimator.hpp"

#include <exception>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"

namespace ibpc
{

//==================================================================================================
PoseEstimator::PoseEstimator(const rclcpp::NodeOptions & options)
: Node("ibpc_pose_estimator", options)
{
  RCLCPP_INFO(
      this->get_logger(),
      "Starting ibpc_pose_estimator..."
  );
  // Get the path to the model.
  std::string path_str = this->declare_parameter("model_dir", "");
  if (path_str.empty()) {
    throw std::runtime_error("ROS param model_dir cannot be empty!");
  }
  RCLCPP_INFO(
      this->get_logger(),
      "Loading model from path [ %s ].",
      path_str.c_str()
  );
  model_dir_ = std::filesystem::path(std::move(path_str));

  std::string srv_name =
    this->declare_parameter("service_name", "/get_pose_estimates");
  RCLCPP_INFO(
      this->get_logger(),
      "Pose estimates can be queried at %s.",
      srv_name.c_str()
  );
  srv_ = this->create_service<GetPoseEstimates>(
      std::move(srv_name),
    [this](std::shared_ptr<const GetPoseEstimates::Request> request,
    std::shared_ptr<GetPoseEstimates::Response> response)
    {
      response->pose_estimates = this->get_pose_estimates(std::move(request));
      }
  );
}

//==================================================================================================
const std::filesystem::path & PoseEstimator::model_dir() const
{
  return model_dir_;
}

//==================================================================================================
auto PoseEstimator::get_pose_estimates(
  std::shared_ptr<const GetPoseEstimates::Request> request) -> std::vector<PoseEstimate>
{
  std::vector<PoseEstimate> pose_estimates = {};

  // Fill.

  return pose_estimates;
}

}  // namespace ibpc

//==================================================================================================
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ibpc::PoseEstimator)
