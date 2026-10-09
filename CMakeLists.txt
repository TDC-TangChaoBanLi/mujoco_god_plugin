cmake_minimum_required(VERSION 3.16)
project(mujoco_god_plugin)

# ---- dependencies ----------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(mujoco_vendor REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)

# ---- interfaces ------------------------------------------------------------
set(msg_files
  msg/ActuatorState.msg
  msg/ActuatorStateArray.msg
  msg/Contact.msg
)

set(srv_files
  srv/GetObjectPose.srv
  srv/SetObjectPose.srv
  srv/GetJointStates.srv
  srv/GetContacts.srv
  srv/GetActuatorState.srv
  srv/SetActuatorControl.srv
)

set(action_files
  action/SetObjectPose.action
  action/SetActuatorControl.action
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    sensor_msgs
    std_msgs
)

# ---- plugin library -------------------------------------------------------
add_library(${PROJECT_NAME}_impl SHARED
  src/god_plugin.cpp
)

target_include_directories(${PROJECT_NAME}_impl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${mujoco_vendor_INCLUDE_DIRS}
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(${PROJECT_NAME}_impl
  mujoco_ros2_control_plugins::mujoco_ros2_control_plugins
  ${cpp_typesupport_target}
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  pluginlib::pluginlib
  ${sensor_msgs_TARGETS}
  ${std_srvs_TARGETS}
  mujoco::mujoco
)

# MuJoCo statically links TinyXML2 which conflicts with the system TinyXML2 used by
# FastDDS; make sure the system one is loaded first.
find_package(tinyxml2_vendor QUIET)
find_package(tinyxml2 QUIET)
if(TARGET tinyxml2::tinyxml2)
  target_link_options(${PROJECT_NAME}_impl PUBLIC
    "LINKER:--push-state,--no-as-needed"
    "$<TARGET_LINKER_FILE:tinyxml2::tinyxml2>"
    "LINKER:--pop-state"
  )
endif()

install(
  TARGETS ${PROJECT_NAME}_impl
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins mujoco_god_plugin.xml)

ament_export_include_directories(include)
ament_export_dependencies(rclcpp rclcpp_action pluginlib mujoco_vendor mujoco_ros2_control_plugins)
ament_export_libraries(${PROJECT_NAME}_impl)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
