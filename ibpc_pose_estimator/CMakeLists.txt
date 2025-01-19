cmake_minimum_required(VERSION 3.8)
project(ibpc_pose_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find packages.
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ibpc_interfaces REQUIRED)
find_package(image_geometry REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

#===============================================================================
add_library(ibpc_pose_estimator_component SHARED src/ibpc_pose_estimator.cpp)

target_link_libraries(ibpc_pose_estimator_component
  PUBLIC
    cv_bridge::cv_bridge
    image_geometry::image_geometry
    rclcpp::rclcpp
    rclcpp_components::component
    ${ibpc_interfaces_TARGETS}
)

target_compile_features(ibpc_pose_estimator_component INTERFACE cxx_std_17)

rclcpp_components_register_node(ibpc_pose_estimator_component
  PLUGIN "ibpc::PoseEstimator"
  EXECUTABLE ibpc_pose_estimator)


#===============================================================================
install(
  TARGETS ibpc_pose_estimator_component
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

ament_package()
