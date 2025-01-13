// TODO: Copyright.

#ifndef IPBC_POSE_ESTIMATOR__IBPC_POSE_ESTIMATOR_HPP_
#define IPBC_POSE_ESTIMATOR__IBPC_POSE_ESTIMATOR_HPP_

#include "rclcpp/rclcpp.hpp"

#include "ibpc_interfaces/msg/pose_estimate.hpp"
#include "ibpc_interfaces/srv/get_pose_estimates.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace ibpc {

//==================================================================================================
class PoseEstimator : public rclcpp::Node
{
public:
  using GetPoseEstimates = ibpc_interfaces::srv::GetPoseEstimates;
  using PoseEstimate = ibpc_interfaces::msg::PoseEstimate;

  PoseEstimator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  bool load_model(const std::filesystem::path & model_path);

  // TODO(Yadunund): Replace request with concrete args that participants will be familiar wiht.
  std::vector<PoseEstimate> get_pose_estimates(
    std::shared_ptr<const GetPoseEstimates::Request> request);

private:
  std::shared_ptr<rclcpp::Service<GetPoseEstimates>> srv_;
};

}  // namespace ibpc

#endif  // IPBC_POSE_ESTIMATOR__IBPC_POSE_ESTIMATOR_HPP_