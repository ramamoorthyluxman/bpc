// TODO: Copyright.

#include "ibpc_pose_estimator.hpp"

#include "rclcpp/rclcpp.hpp"

#include <exception>

namespace ibpc {

//==================================================================================================
PoseEstimator::PoseEstimator(const rclcpp::NodeOptions & options)
: Node("ibpc_pose_estimator", options)
{
	RCLCPP_INFO(
		this->get_logger(),
		"Starting ibpc_pose_estimator..."
	);
	// Get the path to the model.
	std::string path_str = this->declare_parameter("model_path", "");
	if (path_str.empty()) {
		throw std::runtime_error("ROS param model_path cannot be empty!");
	}
	RCLCPP_INFO(
		this->get_logger(),
		"Loading model from path [ %s ].",
		path_str.c_str()
	);
	std::filesystem::path model_path{std::move(path_str)};
	// Load the model.
	if (!this->load_model(model_path)) {
		throw std::runtime_error("Failed to load model.");
	} else {
		RCLCPP_INFO(
			this->get_logger(),
			"Model successfully loaded!"
		);
	}
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
bool PoseEstimator::load_model(const std::filesystem::path & model_path)
{
	// Fill.
	return true;
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