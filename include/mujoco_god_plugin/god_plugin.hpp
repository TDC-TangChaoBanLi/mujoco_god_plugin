// Copyright 2026 TDC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_GOD_PLUGIN__GOD_PLUGIN_HPP_
#define MUJOCO_GOD_PLUGIN__GOD_PLUGIN_HPP_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "mujoco_god_plugin/action/set_actuator_control.hpp"
#include "mujoco_god_plugin/action/set_object_pose.hpp"
#include "mujoco_god_plugin/msg/actuator_state_array.hpp"
#include "mujoco_god_plugin/srv/get_actuator_state.hpp"
#include "mujoco_god_plugin/srv/get_joint_states.hpp"
#include "mujoco_god_plugin/srv/get_contacts.hpp"
#include "mujoco_god_plugin/srv/get_object_pose.hpp"
#include "mujoco_god_plugin/srv/set_actuator_control.hpp"
#include "mujoco_god_plugin/srv/set_object_pose.hpp"
#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_god_plugin
{

/**
 * @brief "God mode" plugin: read and write arbitrary MuJoCo simulation state.
 *
 * Reads (topics + services)
 * -------------------------
 *  * `~/pose/<type>/<name>` (`geometry_msgs/PoseStamped`) for every object listed in the
 *    `objects` parameter, and the `~/get_object_pose` service,
 *  * `~/joint_states` (`sensor_msgs/JointState`) and the `~/get_joint_states` service,
 *  * `~/actuator_state` (`mujoco_god_plugin/ActuatorStateArray`) and the
 *    `~/get_actuator_state` service.
 *
 * Writes (services + actions)
 * ---------------------------
 *  * `~/set_object_pose`: only objects belonging to a body driven by a free joint can be
 *    moved.  For a site/geom the owning free body's pose is back-computed so that the
 *    site/geom lands exactly on the requested pose.  The action variant interpolates the
 *    pose over the requested duration and reports progress as feedback.
 *  * `~/set_actuator_control`: writes `mjData.ctrl`.  With `hold` the value is re-applied
 *    on every physics step so it wins over the ros2_control controllers; otherwise it is
 *    applied for a single step.  The action variant ramps the value over a duration.
 *  * `~/clear_actuator_control` (`std_srvs/Trigger`): drop every held override.
 *
 * Threading
 * ---------
 * Read services answer from a cached snapshot that `update()` refreshes.  All writes go
 * through a queue that `pre_step()` applies to the live `mjData` on the physics thread,
 * which is the only place where they are not immediately overwritten by the hardware
 * interface.
 */
class GodPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  using SetObjectPose = mujoco_god_plugin::action::SetObjectPose;
  using SetActuatorControl = mujoco_god_plugin::action::SetActuatorControl;
  using GoalHandleSetObjectPose = rclcpp_action::ServerGoalHandle<SetObjectPose>;
  using GoalHandleSetActuatorControl = rclcpp_action::ServerGoalHandle<SetActuatorControl>;

  GodPlugin() = default;
  ~GodPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void update(const mjModel* model, mjData* data) override;
  void pre_step(mjData* data) override;
  void on_reset(mjData* data) override;
  void cleanup() override;

private:
  enum class ObjType
  {
    BODY,
    SITE,
    GEOM
  };

  struct TrackedObject
  {
    ObjType type{ ObjType::BODY };
    int id{ -1 };
    std::string name;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub;
  };

  /// A queued absolute pose write for a free body.
  struct PoseWrite
  {
    int body{ -1 };
    double pos[3]{ 0, 0, 0 };
    double quat[4]{ 1, 0, 0, 0 };
  };

  /// An active smooth pose interpolation.
  struct PoseInterp
  {
    int body{ -1 };
    std::string type;
    std::string name;
    double from_pos[3]{ 0, 0, 0 };
    double from_quat[4]{ 1, 0, 0, 0 };
    double to_pos[3]{ 0, 0, 0 };
    double to_quat[4]{ 1, 0, 0, 0 };
    double t0{ 0.0 };
    double duration{ 0.0 };
    std::shared_ptr<GoalHandleSetObjectPose> goal;
  };

  /// An active actuator ramp.
  struct ActuatorRamp
  {
    std::vector<int> ids;
    std::vector<double> from;
    std::vector<double> to;
    bool hold{ true };
    double t0{ 0.0 };
    double duration{ 0.0 };
    std::shared_ptr<GoalHandleSetActuatorControl> goal;
  };

  // ------------------------------------------------------------------- setup
  bool loadParameters();
  static bool parseType(const std::string& s, ObjType& out);
  static const char* typeName(ObjType t);

  // ------------------------------------------------------------------ helpers
  int resolveObject(ObjType type, const std::string& name) const;
  int owningBody(ObjType type, int id) const;
  int freeJointOfBody(int body) const;
  bool backComputeBodyPose(ObjType type, int id, const double* obj_pos, const double* obj_quat, double* body_pos,
                           double* body_quat) const;
  static void quatMul(const double* a, const double* b, double* out);
  static void quatConj(const double* q, double* out);
  static void quatRotate(const double* q, const double* v, double* out);
  static void quatSlerp(const double* a, const double* b, double t, double* out);
  static void matToQuat(const double* m, double* q);

  // ---------------------------------------------------------------- ROS in/out
  using GetObjectPoseSrv = mujoco_god_plugin::srv::GetObjectPose;
  using SetObjectPoseSrv = mujoco_god_plugin::srv::SetObjectPose;
  using GetJointStatesSrv = mujoco_god_plugin::srv::GetJointStates;
  using GetContactsSrv = mujoco_god_plugin::srv::GetContacts;
  using GetActuatorStateSrv = mujoco_god_plugin::srv::GetActuatorState;
  using SetActuatorControlSrv = mujoco_god_plugin::srv::SetActuatorControl;
  using ActuatorStateArrayMsg = mujoco_god_plugin::msg::ActuatorStateArray;
  using ActuatorStateMsg = mujoco_god_plugin::msg::ActuatorState;

  void handleGetObjectPose(const GetObjectPoseSrv::Request::SharedPtr req, GetObjectPoseSrv::Response::SharedPtr res);
  void handleSetObjectPose(const SetObjectPoseSrv::Request::SharedPtr req, SetObjectPoseSrv::Response::SharedPtr res);
  void handleGetJointStates(const GetJointStatesSrv::Request::SharedPtr req,
                            GetJointStatesSrv::Response::SharedPtr res);
  void handleGetContacts(const GetContactsSrv::Request::SharedPtr req,
                         GetContactsSrv::Response::SharedPtr res);
  void handleGetActuatorState(const GetActuatorStateSrv::Request::SharedPtr req,
                              GetActuatorStateSrv::Response::SharedPtr res);
  void handleSetActuatorControl(const SetActuatorControlSrv::Request::SharedPtr req,
                                SetActuatorControlSrv::Response::SharedPtr res);
  void handleClearActuatorControl(const std_srvs::srv::Trigger::Request::SharedPtr req,
                                  std_srvs::srv::Trigger::Response::SharedPtr res);

  rclcpp_action::GoalResponse handleSetObjectPoseGoal(const rclcpp_action::GoalUUID&,
                                                      std::shared_ptr<const SetObjectPose::Goal> goal);
  rclcpp_action::CancelResponse handleSetObjectPoseCancel(const std::shared_ptr<GoalHandleSetObjectPose>);
  void handleSetObjectPoseAccepted(const std::shared_ptr<GoalHandleSetObjectPose> goal);
  rclcpp_action::GoalResponse handleSetActuatorControlGoal(const rclcpp_action::GoalUUID&,
                                                           std::shared_ptr<const SetActuatorControl::Goal> goal);
  rclcpp_action::CancelResponse handleSetActuatorControlCancel(const std::shared_ptr<GoalHandleSetActuatorControl>);
  void handleSetActuatorControlAccepted(const std::shared_ptr<GoalHandleSetActuatorControl> goal);

  // ------------------------------------------------------------------ building
  void buildJointState(const mjData* data, sensor_msgs::msg::JointState& out) const;
  void buildActuatorState(const mjData* data, ActuatorStateArrayMsg& out);
  void fillObjectPose(ObjType type, int id, const std::string& frame_id, geometry_msgs::msg::PoseStamped& out,
                      bool& ok) const;

  // -------------------------------------------------------------------- state
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("GodPlugin") };
  const mjModel* model_{ nullptr };

  double publish_rate_{ 50.0 };
  std::string frame_id_;
  std::string joint_topic_{ "joint_states" };
  std::string actuator_topic_{ "actuator_state" };
  std::vector<std::string> object_specs_;
  std::vector<TrackedObject> objects_;
  std::vector<int> joint_ids_;  ///< 1-DoF joints reported by ~/joint_states

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<ActuatorStateArrayMsg>::SharedPtr actuator_pub_;

  rclcpp::Service<GetObjectPoseSrv>::SharedPtr get_pose_srv_;
  rclcpp::Service<SetObjectPoseSrv>::SharedPtr set_pose_srv_;
  rclcpp::Service<GetJointStatesSrv>::SharedPtr get_joints_srv_;
  rclcpp::Service<GetContactsSrv>::SharedPtr get_contacts_srv_;
  rclcpp::Service<GetActuatorStateSrv>::SharedPtr get_actuators_srv_;
  rclcpp::Service<SetActuatorControlSrv>::SharedPtr set_actuators_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_actuators_srv_;
  rclcpp_action::Server<SetObjectPose>::SharedPtr set_pose_action_;
  rclcpp_action::Server<SetActuatorControl>::SharedPtr set_actuator_action_;

  rclcpp::Time last_publish_{ 0, 0, RCL_ROS_TIME };
  bool last_publish_valid_{ false };

  // Cached snapshot refreshed by update(), answered by the read services.
  mutable std::mutex snapshot_mutex_;
  std::vector<double> body_poses_;  ///< nbody x 7 (pos + quat)
  std::vector<double> site_poses_;  ///< nsite x 7
  std::vector<double> geom_poses_;  ///< ngeom x 7
  std::vector<double> ctrl_values_; ///< nu
  std::vector<double> actuator_force_;
  sensor_msgs::msg::JointState joint_state_;
  ActuatorStateArrayMsg actuator_state_;
  std::vector<mujoco_god_plugin::msg::Contact> contacts_;
  bool snapshot_valid_{ false };

  // Queued writes, applied in pre_step().
  std::mutex write_mutex_;
  std::vector<PoseWrite> pose_writes_;
  std::map<int, double> actuator_override_;  ///< held overrides
  std::map<int, double> actuator_once_;      ///< applied for exactly one step
  std::atomic_bool override_dirty_{ false };

  // Interpolations, advanced in update().
  std::mutex interp_mutex_;
  std::vector<PoseInterp> pose_interps_;
  std::vector<ActuatorRamp> actuator_ramps_;
};

}  // namespace mujoco_god_plugin

#endif  // MUJOCO_GOD_PLUGIN__GOD_PLUGIN_HPP_
