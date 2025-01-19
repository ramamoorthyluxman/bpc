// TODO(Yadunund): Copyright.

#ifndef IBPC_POSE_ESTIMATOR_HPP_
#define IBPC_POSE_ESTIMATOR_HPP_

#include <filesystem>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "ibpc_interfaces/msg/pose_estimate.hpp"
#include "ibpc_interfaces/srv/get_pose_estimates.hpp"

namespace ibpc
{

//==================================================================================================
class PoseEstimator : public rclcpp::Node
{
public:
  using GetPoseEstimates = ibpc_interfaces::srv::GetPoseEstimates;
  using PoseEstimate = ibpc_interfaces::msg::PoseEstimate;

  /// Constructor.
  explicit PoseEstimator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // Get the path to the directory in the local filesystem that contains various models
  // and other files required for inference.
  const std::filesystem::path & model_dir() const;

  // TODO(Yadunund): Replace request with concrete args that participants will be familiar wiht.
  std::vector<PoseEstimate> get_pose_estimates(
    std::shared_ptr<const GetPoseEstimates::Request> request);

private:
  std::filesystem::path model_dir_;
  std::shared_ptr<rclcpp::Service<GetPoseEstimates>> srv_;
};

}  // namespace ibpc

#endif  // IBPC_POSE_ESTIMATOR_HPP_
