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

#include "mujoco_god_plugin/god_plugin.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

#include <pluginlib/class_list_macros.hpp>

namespace mujoco_god_plugin
{

namespace
{
constexpr double kQuatNormEps = 1e-9;

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

rclcpp::Time toRosTime(double seconds)
{
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9), RCL_ROS_TIME);
}
}  // namespace

// ============================================================================
// math helpers
// ============================================================================

void GodPlugin::quatMul(const double* a, const double* b, double* out)
{
  const double w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  const double x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  const double y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  const double z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  out[0] = w;
  out[1] = x;
  out[2] = y;
  out[3] = z;
}

void GodPlugin::quatConj(const double* q, double* out)
{
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
}

void GodPlugin::quatRotate(const double* q, const double* v, double* out)
{
  const double w = q[0];
  const double tx = 2.0 * (q[2] * v[2] - q[3] * v[1]);
  const double ty = 2.0 * (q[3] * v[0] - q[1] * v[2]);
  const double tz = 2.0 * (q[1] * v[1] - q[2] * v[0]);
  out[0] = v[0] + w * tx + (q[2] * tz - q[3] * ty);
  out[1] = v[1] + w * ty + (q[3] * tx - q[1] * tz);
  out[2] = v[2] + w * tz + (q[1] * ty - q[2] * tx);
}

void GodPlugin::quatSlerp(const double* a, const double* b, double t, double* out)
{
  double bb[4] = { b[0], b[1], b[2], b[3] };
  double dot = a[0] * bb[0] + a[1] * bb[1] + a[2] * bb[2] + a[3] * bb[3];
  if (dot < 0.0)
  {
    dot = -dot;
    for (int i = 0; i < 4; ++i)
    {
      bb[i] = -bb[i];
    }
  }
  if (dot > 0.9995)
  {
    for (int i = 0; i < 4; ++i)
    {
      out[i] = a[i] + t * (bb[i] - a[i]);
    }
  }
  else
  {
    const double theta0 = std::acos(std::min(1.0, dot));
    const double theta = theta0 * t;
    const double s0 = std::sin(theta0 - theta) / std::sin(theta0);
    const double s1 = std::sin(theta) / std::sin(theta0);
    for (int i = 0; i < 4; ++i)
    {
      out[i] = s0 * a[i] + s1 * bb[i];
    }
  }
  const double n = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
  if (n > kQuatNormEps)
  {
    for (int i = 0; i < 4; ++i)
    {
      out[i] /= n;
    }
  }
}

void GodPlugin::matToQuat(const double* m, double* q)
{
  const double tr = m[0] + m[4] + m[8];
  if (tr > 0.0)
  {
    const double s = std::sqrt(tr + 1.0) * 2.0;
    q[0] = 0.25 * s;
    q[1] = (m[7] - m[5]) / s;
    q[2] = (m[2] - m[6]) / s;
    q[3] = (m[3] - m[1]) / s;
  }
  else if (m[0] > m[4] && m[0] > m[8])
  {
    const double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
    q[0] = (m[7] - m[5]) / s;
    q[1] = 0.25 * s;
    q[2] = (m[1] + m[3]) / s;
    q[3] = (m[2] + m[6]) / s;
  }
  else if (m[4] > m[8])
  {
    const double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
    q[0] = (m[2] - m[6]) / s;
    q[1] = (m[1] + m[3]) / s;
    q[2] = 0.25 * s;
    q[3] = (m[5] + m[7]) / s;
  }
  else
  {
    const double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
    q[0] = (m[3] - m[1]) / s;
    q[1] = (m[2] + m[6]) / s;
    q[2] = (m[5] + m[7]) / s;
    q[3] = 0.25 * s;
  }
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n > kQuatNormEps)
  {
    for (int i = 0; i < 4; ++i)
    {
      q[i] /= n;
    }
  }
  else
  {
    q[0] = 1.0;
    q[1] = q[2] = q[3] = 0.0;
  }
}

// ============================================================================
// type helpers
// ============================================================================

bool GodPlugin::parseType(const std::string& s, ObjType& out)
{
  const std::string t = lower(s);
  if (t == "body")
  {
    out = ObjType::BODY;
    return true;
  }
  if (t == "site")
  {
    out = ObjType::SITE;
    return true;
  }
  if (t == "geom" || t == "geometry")
  {
    out = ObjType::GEOM;
    return true;
  }
  return false;
}

const char* GodPlugin::typeName(ObjType t)
{
  switch (t)
  {
    case ObjType::BODY:
      return "body";
    case ObjType::SITE:
      return "site";
    case ObjType::GEOM:
      return "geom";
  }
  return "?";
}

int GodPlugin::resolveObject(ObjType type, const std::string& name) const
{
  switch (type)
  {
    case ObjType::BODY:
      return mj_name2id(model_, mjOBJ_BODY, name.c_str());
    case ObjType::SITE:
      return mj_name2id(model_, mjOBJ_SITE, name.c_str());
    case ObjType::GEOM:
      return mj_name2id(model_, mjOBJ_GEOM, name.c_str());
  }
  return -1;
}

int GodPlugin::owningBody(ObjType type, int id) const
{
  switch (type)
  {
    case ObjType::BODY:
      return id;
    case ObjType::SITE:
      return model_->site_bodyid[id];
    case ObjType::GEOM:
      return model_->geom_bodyid[id];
  }
  return -1;
}

int GodPlugin::freeJointOfBody(int body) const
{
  for (int j = 0; j < model_->njnt; ++j)
  {
    if (model_->jnt_bodyid[j] == body && model_->jnt_type[j] == mjJNT_FREE)
    {
      return j;
    }
  }
  return -1;
}

bool GodPlugin::backComputeBodyPose(ObjType type, int id, const double* obj_pos, const double* obj_quat,
                                    double* body_pos, double* body_quat) const
{
  // x_obj = x_body o L  =>  x_body = x_obj o L^-1   (L = object frame inside the body)
  if (type == ObjType::BODY)
  {
    std::copy(obj_pos, obj_pos + 3, body_pos);
    std::copy(obj_quat, obj_quat + 4, body_quat);
    return true;
  }

  const double* l_pos = (type == ObjType::SITE) ? (model_->site_pos + 3 * id) : (model_->geom_pos + 3 * id);
  const double* l_quat = (type == ObjType::SITE) ? (model_->site_quat + 4 * id) : (model_->geom_quat + 4 * id);

  double l_quat_c[4];
  quatConj(l_quat, l_quat_c);
  quatMul(obj_quat, l_quat_c, body_quat);
  double rl[3];
  quatRotate(body_quat, l_pos, rl);
  for (int i = 0; i < 3; ++i)
  {
    body_pos[i] = obj_pos[i] - rl[i];
  }
  return true;
}

// ============================================================================
// init
// ============================================================================

bool GodPlugin::loadParameters()
{
  const std::string prefix = "mujoco_plugins." + node_->get_sub_namespace() + ".";
  auto declare = [&](const std::string& n, auto def) {
    const std::string full = prefix + n;
    if (!node_->has_parameter(full))
    {
      node_->declare_parameter(full, def);
    }
    return full;
  };

  publish_rate_ = node_->get_parameter(declare("publish_rate", 50.0)).as_double();
  if (publish_rate_ <= 0.0)
  {
    RCLCPP_ERROR(logger_, "publish_rate must be > 0 (got %f).", publish_rate_);
    return false;
  }
  frame_id_ = node_->get_parameter(declare("frame_id", std::string(""))).as_string();
  joint_topic_ = node_->get_parameter(declare("joint_states_topic", std::string("joint_states"))).as_string();
  actuator_topic_ = node_->get_parameter(declare("actuator_state_topic", std::string("actuator_state"))).as_string();
  object_specs_ = node_->get_parameter(declare("objects", std::vector<std::string>{})).as_string_array();
  return true;
}

bool GodPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* /*data*/)
{
  node_ = node;
  logger_ = node_->get_logger().get_child(node_->get_sub_namespace());
  model_ = model;

  if (!loadParameters())
  {
    return false;
  }

  joint_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(joint_topic_, rclcpp::SystemDefaultsQoS());
  actuator_pub_ = node_->create_publisher<ActuatorStateArrayMsg>(actuator_topic_, rclcpp::SystemDefaultsQoS());

  for (const auto& spec : object_specs_)
  {
    const auto colon = spec.find(':');
    if (colon == std::string::npos)
    {
      RCLCPP_WARN(logger_, "objects entry '%s' is not of the form '<body|site|geom>:<name>'; skipping.", spec.c_str());
      continue;
    }
    ObjType type;
    const std::string type_str = spec.substr(0, colon);
    const std::string name = spec.substr(colon + 1);
    if (!parseType(type_str, type))
    {
      RCLCPP_WARN(logger_, "objects entry '%s' has an unknown type '%s'; skipping.", spec.c_str(), type_str.c_str());
      continue;
    }
    const int id = resolveObject(type, name);
    if (id < 0)
    {
      RCLCPP_WARN(logger_, "objects entry '%s': '%s' is not a %s of this model; skipping.", spec.c_str(), name.c_str(),
                  typeName(type));
      continue;
    }
    TrackedObject o;
    o.type = type;
    o.id = id;
    o.name = name;
    o.pub = node_->create_publisher<geometry_msgs::msg::PoseStamped>("pose/" + std::string(typeName(type)) + "/" + name,
                                                                    rclcpp::SystemDefaultsQoS());
    objects_.push_back(std::move(o));
  }

  for (int j = 0; j < model_->njnt; ++j)
  {
    if (model_->jnt_type[j] == mjJNT_HINGE || model_->jnt_type[j] == mjJNT_SLIDE)
    {
      joint_ids_.push_back(j);
    }
  }

  body_poses_.assign(static_cast<size_t>(model_->nbody) * 7, 0.0);
  site_poses_.assign(static_cast<size_t>(model_->nsite) * 7, 0.0);
  geom_poses_.assign(static_cast<size_t>(model_->ngeom) * 7, 0.0);
  ctrl_values_.assign(static_cast<size_t>(model_->nu), 0.0);
  actuator_force_.assign(static_cast<size_t>(model_->nu), 0.0);
  for (int i = 0; i < model_->nbody; ++i)
  {
    body_poses_[7 * i + 3] = 1.0;
  }
  for (int i = 0; i < model_->nsite; ++i)
  {
    site_poses_[7 * i + 3] = 1.0;
  }
  for (int i = 0; i < model_->ngeom; ++i)
  {
    geom_poses_[7 * i + 3] = 1.0;
  }

  get_pose_srv_ = node_->create_service<GetObjectPoseSrv>(
      "get_object_pose",
      std::bind(&GodPlugin::handleGetObjectPose, this, std::placeholders::_1, std::placeholders::_2));
  set_pose_srv_ = node_->create_service<SetObjectPoseSrv>(
      "set_object_pose",
      std::bind(&GodPlugin::handleSetObjectPose, this, std::placeholders::_1, std::placeholders::_2));
  get_joints_srv_ = node_->create_service<GetJointStatesSrv>(
      "get_joint_states",
      std::bind(&GodPlugin::handleGetJointStates, this, std::placeholders::_1, std::placeholders::_2));
  get_contacts_srv_ = node_->create_service<GetContactsSrv>(
      "get_contacts",
      std::bind(&GodPlugin::handleGetContacts, this, std::placeholders::_1, std::placeholders::_2));
  get_actuators_srv_ = node_->create_service<GetActuatorStateSrv>(
      "get_actuator_state",
      std::bind(&GodPlugin::handleGetActuatorState, this, std::placeholders::_1, std::placeholders::_2));
  set_actuators_srv_ = node_->create_service<SetActuatorControlSrv>(
      "set_actuator_control",
      std::bind(&GodPlugin::handleSetActuatorControl, this, std::placeholders::_1, std::placeholders::_2));
  clear_actuators_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      "clear_actuator_control",
      std::bind(&GodPlugin::handleClearActuatorControl, this, std::placeholders::_1, std::placeholders::_2));

  set_pose_action_ = rclcpp_action::create_server<SetObjectPose>(
      node_, "set_object_pose",
      std::bind(&GodPlugin::handleSetObjectPoseGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&GodPlugin::handleSetObjectPoseCancel, this, std::placeholders::_1),
      std::bind(&GodPlugin::handleSetObjectPoseAccepted, this, std::placeholders::_1));
  set_actuator_action_ = rclcpp_action::create_server<SetActuatorControl>(
      node_, "set_actuator_control",
      std::bind(&GodPlugin::handleSetActuatorControlGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&GodPlugin::handleSetActuatorControlCancel, this, std::placeholders::_1),
      std::bind(&GodPlugin::handleSetActuatorControlAccepted, this, std::placeholders::_1));

  RCLCPP_INFO(logger_,
              "GodPlugin initialised: %zu tracked object(s), %zu joint(s), %zu actuator(s). Topics '%s', '%s'; "
              "services get/set_object_pose, get_joint_states, get/set_actuator_control.",
              objects_.size(), joint_ids_.size(), static_cast<size_t>(model_->nu),
              joint_pub_->get_topic_name(), actuator_pub_->get_topic_name());
  return true;
}

// ============================================================================
// message building
// ============================================================================

void GodPlugin::buildJointState(const mjData* data, sensor_msgs::msg::JointState& out) const
{
  out.name.clear();
  out.position.clear();
  out.velocity.clear();
  out.effort.clear();
  for (int j : joint_ids_)
  {
    const int qadr = model_->jnt_qposadr[j];
    const int vadr = model_->jnt_dofadr[j];
    const char* n = mj_id2name(model_, mjOBJ_JOINT, j);
    out.name.emplace_back(n ? n : ("joint_" + std::to_string(j)));
    out.position.push_back(data->qpos[qadr]);
    out.velocity.push_back(data->qvel[vadr]);
    out.effort.push_back(data->qfrc_actuator[vadr]);
  }
}

void GodPlugin::buildActuatorState(const mjData* data, ActuatorStateArrayMsg& out)
{
  out.actuators.clear();
  out.actuators.reserve(static_cast<size_t>(model_->nu));
  for (int i = 0; i < model_->nu; ++i)
  {
    ActuatorStateMsg a;
    const char* an = mj_id2name(model_, mjOBJ_ACTUATOR, i);
    a.name = an ? an : ("actuator_" + std::to_string(i));
    a.ctrl = data->ctrl[i];
    a.ctrl_min = model_->actuator_ctrlrange[2 * i];
    a.ctrl_max = model_->actuator_ctrlrange[2 * i + 1];
    a.actuator_force = data->actuator_force[i];
    a.joint_position = std::numeric_limits<double>::quiet_NaN();
    a.joint_velocity = std::numeric_limits<double>::quiet_NaN();
    const int j = model_->actuator_trnid[2 * i];  // first transmission target
    if (j >= 0 && j < model_->njnt &&
        (model_->jnt_type[j] == mjJNT_HINGE || model_->jnt_type[j] == mjJNT_SLIDE))
    {
      a.joint_position = data->qpos[model_->jnt_qposadr[j]];
      a.joint_velocity = data->qvel[model_->jnt_dofadr[j]];
    }
    std::lock_guard<std::mutex> lock(write_mutex_);
    a.overridden = actuator_override_.count(i) > 0;
    out.actuators.push_back(a);
  }
}

void GodPlugin::fillObjectPose(ObjType type, int id, const std::string& ref, geometry_msgs::msg::PoseStamped& out,
                               bool& ok) const
{
  ok = false;
  const std::vector<double>* arr = nullptr;
  switch (type)
  {
    case ObjType::BODY:
      arr = &body_poses_;
      break;
    case ObjType::SITE:
      arr = &site_poses_;
      break;
    case ObjType::GEOM:
      arr = &geom_poses_;
      break;
  }
  if (arr == nullptr || id < 0 || 7 * static_cast<size_t>(id) + 7 > arr->size())
  {
    return;
  }
  double pos[3] = { (*arr)[7 * id], (*arr)[7 * id + 1], (*arr)[7 * id + 2] };
  double quat[4] = { (*arr)[7 * id + 3], (*arr)[7 * id + 4], (*arr)[7 * id + 5], (*arr)[7 * id + 6] };

  out.header.frame_id = frame_id_;
  if (!ref.empty())
  {
    const int fb = mj_name2id(model_, mjOBJ_BODY, ref.c_str());
    if (fb >= 0 && 7 * static_cast<size_t>(fb) + 7 <= body_poses_.size())
    {
      const double* fp = &body_poses_[7 * fb];
      const double* fq = &body_poses_[7 * fb + 3];
      double fqc[4];
      quatConj(fq, fqc);
      const double d[3] = { pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2] };
      double rp[3];
      quatRotate(fqc, d, rp);
      double rq[4];
      quatMul(fqc, quat, rq);
      std::copy(rp, rp + 3, pos);
      std::copy(rq, rq + 4, quat);
      out.header.frame_id = ref;
    }
  }
  out.pose.position.x = pos[0];
  out.pose.position.y = pos[1];
  out.pose.position.z = pos[2];
  out.pose.orientation.w = quat[0];
  out.pose.orientation.x = quat[1];
  out.pose.orientation.y = quat[2];
  out.pose.orientation.z = quat[3];
  ok = true;
}

// ============================================================================
// ROS service callbacks
// ============================================================================

void GodPlugin::handleGetObjectPose(const GetObjectPoseSrv::Request::SharedPtr req,
                                    GetObjectPoseSrv::Response::SharedPtr res)
{
  ObjType type;
  if (!parseType(req->type, type))
  {
    res->success = false;
    res->message = "Unknown object type '" + req->type + "' (expected body/site/geom).";
    return;
  }
  const int id = resolveObject(type, req->name);
  if (id < 0)
  {
    res->success = false;
    res->message = "'" + req->name + "' is not a " + typeName(type) + " of this model.";
    return;
  }
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  geometry_msgs::msg::PoseStamped ps;
  bool ok = false;
  fillObjectPose(type, id, req->frame_id, ps, ok);
  if (!ok)
  {
    res->success = false;
    res->message = "No pose available yet (the plugin has not completed an update cycle).";
    return;
  }
  res->success = true;
  res->message = "ok";
  res->pose = ps;
  res->pose.header.stamp = node_->get_clock()->now();
}

void GodPlugin::handleSetObjectPose(const SetObjectPoseSrv::Request::SharedPtr req,
                                    SetObjectPoseSrv::Response::SharedPtr res)
{
  ObjType type;
  if (!parseType(req->type, type))
  {
    res->success = false;
    res->message = "Unknown object type '" + req->type + "' (expected body/site/geom).";
    return;
  }
  const int id = resolveObject(type, req->name);
  if (id < 0)
  {
    res->success = false;
    res->message = "'" + req->name + "' is not a " + typeName(type) + " of this model.";
    return;
  }
  const int body = owningBody(type, id);
  if (freeJointOfBody(body) < 0)
  {
    res->success = false;
    res->message = std::string("Cannot move ") + typeName(type) + " '" + req->name +
                   "': its body is not driven by a free joint.";
    return;
  }

  // resolve the reference frame
  double obj_pos[3] = { req->pose.pose.position.x, req->pose.pose.position.y, req->pose.pose.position.z };
  double obj_quat[4] = { req->pose.pose.orientation.w, req->pose.pose.orientation.x, req->pose.pose.orientation.y,
                         req->pose.pose.orientation.z };
  const std::string ref = req->pose.header.frame_id;
  if (!ref.empty())
  {
    const int fb = mj_name2id(model_, mjOBJ_BODY, ref.c_str());
    if (fb < 0)
    {
      res->success = false;
      res->message = "Unknown reference frame body '" + ref + "'.";
      return;
    }
    double fp[3], fq[4];
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      if (7 * static_cast<size_t>(fb) + 7 > body_poses_.size())
      {
        res->success = false;
        res->message = "Reference frame pose is not available yet.";
        return;
      }
      std::copy(&body_poses_[7 * fb], &body_poses_[7 * fb] + 3, fp);
      std::copy(&body_poses_[7 * fb + 3], &body_poses_[7 * fb + 3] + 4, fq);
    }
    // x_world = x_frame o x_ref
    double rp[3];
    quatRotate(fq, obj_pos, rp);
    double wq[4];
    quatMul(fq, obj_quat, wq);
    for (int i = 0; i < 3; ++i)
    {
      obj_pos[i] = rp[i] + fp[i];
    }
    std::copy(wq, wq + 4, obj_quat);
  }

  double body_pos[3], body_quat[4];
  backComputeBodyPose(type, id, obj_pos, obj_quat, body_pos, body_quat);

  PoseWrite pw;
  pw.body = body;
  std::copy(body_pos, body_pos + 3, pw.pos);
  std::copy(body_quat, body_quat + 4, pw.quat);
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    pose_writes_.push_back(pw);
  }
  res->success = true;
  res->message = "Pose queued (applied on the next physics step).";
}

void GodPlugin::handleGetContacts(const GetContactsSrv::Request::SharedPtr,
                                 GetContactsSrv::Response::SharedPtr res)
{
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  res->success = snapshot_valid_;
  res->message = snapshot_valid_ ? "ok" : "simulator snapshot unavailable";
  res->header = joint_state_.header;
  if (snapshot_valid_) res->contacts = contacts_;
}

void GodPlugin::handleGetJointStates(const GetJointStatesSrv::Request::SharedPtr req,
                                     GetJointStatesSrv::Response::SharedPtr res)
{
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  if (!snapshot_valid_)
  {
    res->success = false;
    res->message = "No snapshot available yet.";
    return;
  }
  res->state = joint_state_;
  if (!req->names.empty())
  {
    sensor_msgs::msg::JointState filtered;
    filtered.header = joint_state_.header;
    for (const auto& n : req->names)
    {
      auto it = std::find(joint_state_.name.begin(), joint_state_.name.end(), n);
      if (it == joint_state_.name.end())
      {
        res->success = false;
        res->message = "Joint '" + n + "' is unknown or not a 1-DoF joint.";
        return;
      }
      const size_t k = static_cast<size_t>(std::distance(joint_state_.name.begin(), it));
      filtered.name.push_back(joint_state_.name[k]);
      filtered.position.push_back(joint_state_.position[k]);
      filtered.velocity.push_back(joint_state_.velocity[k]);
      filtered.effort.push_back(joint_state_.effort[k]);
    }
    res->state = filtered;
  }
  res->state.header.stamp = node_->get_clock()->now();
  res->success = true;
  res->message = "ok";
}

void GodPlugin::handleGetActuatorState(const GetActuatorStateSrv::Request::SharedPtr req,
                                       GetActuatorStateSrv::Response::SharedPtr res)
{
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  if (!snapshot_valid_)
  {
    res->success = false;
    res->message = "No snapshot available yet.";
    return;
  }
  res->state = actuator_state_;
  if (!req->names.empty())
  {
    ActuatorStateArrayMsg filtered;
    filtered.header = actuator_state_.header;
    for (const auto& n : req->names)
    {
      auto it = std::find_if(actuator_state_.actuators.begin(), actuator_state_.actuators.end(),
                             [&](const ActuatorStateMsg& a) { return a.name == n; });
      if (it == actuator_state_.actuators.end())
      {
        res->success = false;
        res->message = "Actuator '" + n + "' is unknown.";
        return;
      }
      filtered.actuators.push_back(*it);
    }
    res->state = filtered;
  }
  res->state.header.stamp = node_->get_clock()->now();
  res->success = true;
  res->message = "ok";
}

void GodPlugin::handleSetActuatorControl(const SetActuatorControlSrv::Request::SharedPtr req,
                                         SetActuatorControlSrv::Response::SharedPtr res)
{
  if (req->names.size() != req->values.size())
  {
    res->success = false;
    res->message = "names and values must have the same length.";
    return;
  }
  std::vector<int> ids;
  for (const auto& n : req->names)
  {
    const int i = mj_name2id(model_, mjOBJ_ACTUATOR, n.c_str());
    if (i < 0)
    {
      res->success = false;
      res->message = "Actuator '" + n + "' is unknown.";
      return;
    }
    ids.push_back(i);
  }
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    for (size_t k = 0; k < ids.size(); ++k)
    {
      if (req->hold)
      {
        actuator_override_[ids[k]] = req->values[k];
      }
      else
      {
        actuator_once_[ids[k]] = req->values[k];
      }
    }
    override_dirty_.store(true);
  }
  res->success = true;
  res->message = req->hold ? "Control inputs queued (held)." : "Control inputs queued (single step).";
}

void GodPlugin::handleClearActuatorControl(const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
                                           std_srvs::srv::Trigger::Response::SharedPtr res)
{
  std::lock_guard<std::mutex> lock(write_mutex_);
  actuator_override_.clear();
  override_dirty_.store(true);
  res->success = true;
  res->message = "All actuator overrides cleared.";
}

// ============================================================================
// action callbacks
// ============================================================================

rclcpp_action::GoalResponse GodPlugin::handleSetObjectPoseGoal(const rclcpp_action::GoalUUID&,
                                                               std::shared_ptr<const SetObjectPose::Goal> goal)
{
  ObjType type;
  if (!parseType(goal->type, type))
  {
    return rclcpp_action::GoalResponse::REJECT;
  }
  const int id = resolveObject(type, goal->name);
  if (id < 0 || freeJointOfBody(owningBody(type, id)) < 0)
  {
    RCLCPP_WARN(logger_, "Rejecting set_object_pose goal: '%s' cannot be moved.", goal->name.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse GodPlugin::handleSetObjectPoseCancel(const std::shared_ptr<GoalHandleSetObjectPose>)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void GodPlugin::handleSetObjectPoseAccepted(const std::shared_ptr<GoalHandleSetObjectPose> goal)
{
  ObjType type;
  parseType(goal->get_goal()->type, type);
  const int id = resolveObject(type, goal->get_goal()->name);
  const int body = owningBody(type, id);

  double target_pos[3] = { goal->get_goal()->pose.pose.position.x, goal->get_goal()->pose.pose.position.y,
                           goal->get_goal()->pose.pose.position.z };
  double target_quat[4] = { goal->get_goal()->pose.pose.orientation.w, goal->get_goal()->pose.pose.orientation.x,
                            goal->get_goal()->pose.pose.orientation.y, goal->get_goal()->pose.pose.orientation.z };

  // start pose of the object
  double start_pos[3], start_quat[4];
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    const std::vector<double>* arr =
        (type == ObjType::BODY) ? &body_poses_ : ((type == ObjType::SITE) ? &site_poses_ : &geom_poses_);
    std::copy(&(*arr)[7 * id], &(*arr)[7 * id] + 3, start_pos);
    std::copy(&(*arr)[7 * id + 3], &(*arr)[7 * id + 3] + 4, start_quat);
  }

  // A body target is the body pose; a site/geom target has to be converted into the
  // equivalent body pose.
  double from_body_pos[3], from_body_quat[4];
  double to_body_pos[3], to_body_quat[4];
  if (type == ObjType::BODY)
  {
    std::copy(start_pos, start_pos + 3, from_body_pos);
    std::copy(start_quat, start_quat + 4, from_body_quat);
    std::copy(target_pos, target_pos + 3, to_body_pos);
    std::copy(target_quat, target_quat + 4, to_body_quat);
  }
  else
  {
    backComputeBodyPose(type, id, start_pos, start_quat, from_body_pos, from_body_quat);
    backComputeBodyPose(type, id, target_pos, target_quat, to_body_pos, to_body_quat);
  }

  PoseInterp pi;
  pi.body = body;
  pi.type = typeName(type);
  pi.name = goal->get_goal()->name;
  std::copy(from_body_pos, from_body_pos + 3, pi.from_pos);
  std::copy(from_body_quat, from_body_quat + 4, pi.from_quat);
  std::copy(to_body_pos, to_body_pos + 3, pi.to_pos);
  std::copy(to_body_quat, to_body_quat + 4, pi.to_quat);
  pi.t0 = node_->get_clock()->now().seconds();
  pi.duration = goal->get_goal()->duration;
  pi.goal = goal;
  {
    std::lock_guard<std::mutex> lock(interp_mutex_);
    pose_interps_.push_back(std::move(pi));
  }
}

rclcpp_action::GoalResponse GodPlugin::handleSetActuatorControlGoal(const rclcpp_action::GoalUUID&,
                                                                    std::shared_ptr<const SetActuatorControl::Goal> goal)
{
  if (goal->names.size() != goal->values.size())
  {
    return rclcpp_action::GoalResponse::REJECT;
  }
  for (const auto& n : goal->names)
  {
    if (mj_name2id(model_, mjOBJ_ACTUATOR, n.c_str()) < 0)
    {
      RCLCPP_WARN(logger_, "Rejecting set_actuator_control goal: unknown actuator '%s'.", n.c_str());
      return rclcpp_action::GoalResponse::REJECT;
    }
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse GodPlugin::handleSetActuatorControlCancel(
    const std::shared_ptr<GoalHandleSetActuatorControl>)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void GodPlugin::handleSetActuatorControlAccepted(const std::shared_ptr<GoalHandleSetActuatorControl> goal)
{
  ActuatorRamp ar;
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    for (size_t k = 0; k < goal->get_goal()->names.size(); ++k)
    {
      const int i = mj_name2id(model_, mjOBJ_ACTUATOR, goal->get_goal()->names[k].c_str());
      ar.ids.push_back(i);
      ar.from.push_back(i < static_cast<int>(ctrl_values_.size()) ? ctrl_values_[i] : 0.0);
      ar.to.push_back(goal->get_goal()->values[k]);
    }
  }
  ar.hold = goal->get_goal()->hold;
  ar.t0 = node_->get_clock()->now().seconds();
  ar.duration = goal->get_goal()->ramp_duration;
  ar.goal = goal;
  {
    std::lock_guard<std::mutex> lock(interp_mutex_);
    actuator_ramps_.push_back(std::move(ar));
  }
}

// ============================================================================
// update()
// ============================================================================

void GodPlugin::update(const mjModel* /*model*/, mjData* data)
{
  const rclcpp::Time now = node_->get_clock()->now();

  // ---- 1. advance smooth pose interpolations and actuator ramps -------------
  {
    std::lock_guard<std::mutex> lock(interp_mutex_);

    for (auto it = pose_interps_.begin(); it != pose_interps_.end();)
    {
      PoseInterp& pi = *it;
      const double elapsed = (now - toRosTime(pi.t0)).seconds();
      const double u = pi.duration > 0.0 ? std::min(1.0, std::max(0.0, elapsed / pi.duration)) : 1.0;

      double pos[3];
      double quat[4];
      for (int i = 0; i < 3; ++i)
      {
        pos[i] = pi.from_pos[i] + u * (pi.to_pos[i] - pi.from_pos[i]);
      }
      quatSlerp(pi.from_quat, pi.to_quat, u, quat);

      {
        std::lock_guard<std::mutex> wlock(write_mutex_);
        PoseWrite pw;
        pw.body = pi.body;
        std::copy(pos, pos + 3, pw.pos);
        std::copy(quat, quat + 4, pw.quat);
        pose_writes_.push_back(pw);
      }

      if (pi.goal && pi.goal->is_active())
      {
        auto fb = std::make_shared<SetObjectPose::Feedback>();
        fb->progress = u;
        fb->current.header.stamp = now;
        fb->current.header.frame_id = frame_id_;
        fb->current.pose.position.x = pos[0];
        fb->current.pose.position.y = pos[1];
        fb->current.pose.position.z = pos[2];
        fb->current.pose.orientation.w = quat[0];
        fb->current.pose.orientation.x = quat[1];
        fb->current.pose.orientation.y = quat[2];
        fb->current.pose.orientation.z = quat[3];
        pi.goal->publish_feedback(fb);
      }

      if (u >= 1.0)
      {
        if (pi.goal && pi.goal->is_active())
        {
          auto res = std::make_shared<SetObjectPose::Result>();
          res->success = true;
          res->message = "Pose reached.";
          pi.goal->succeed(res);
        }
        it = pose_interps_.erase(it);
      }
      else
      {
        ++it;
      }
    }

    for (auto it = actuator_ramps_.begin(); it != actuator_ramps_.end();)
    {
      ActuatorRamp& ar = *it;
      const double elapsed = (now - toRosTime(ar.t0)).seconds();
      const double u = ar.duration > 0.0 ? std::min(1.0, std::max(0.0, elapsed / ar.duration)) : 1.0;
      {
        std::lock_guard<std::mutex> wlock(write_mutex_);
        for (size_t k = 0; k < ar.ids.size(); ++k)
        {
          const double v = ar.from[k] + u * (ar.to[k] - ar.from[k]);
          actuator_override_[ar.ids[k]] = v;
        }
        override_dirty_.store(true);
      }
      if (ar.goal && ar.goal->is_active())
      {
        auto fb = std::make_shared<SetActuatorControl::Feedback>();
        fb->progress = u;
        {
          std::lock_guard<std::mutex> slock(snapshot_mutex_);
          fb->current = actuator_state_;
        }
        ar.goal->publish_feedback(fb);
      }
      if (u >= 1.0)
      {
        if (!ar.hold)
        {
          std::lock_guard<std::mutex> wlock(write_mutex_);
          for (int i : ar.ids)
          {
            actuator_override_.erase(i);
          }
        }
        if (ar.goal && ar.goal->is_active())
        {
          auto res = std::make_shared<SetActuatorControl::Result>();
          res->success = true;
          res->message = ar.hold ? "Ramp finished; control inputs held." : "Ramp finished.";
          ar.goal->succeed(res);
        }
        it = actuator_ramps_.erase(it);
      }
      else
      {
        ++it;
      }
    }
  }

  // ---- 2. refresh the read snapshot and publish -----------------------------
  if (last_publish_valid_ && (now - last_publish_).seconds() < 1.0 / publish_rate_)
  {
    return;
  }
  last_publish_ = now;
  last_publish_valid_ = true;

  std::vector<double> body_poses(static_cast<size_t>(model_->nbody) * 7);
  std::vector<double> site_poses(static_cast<size_t>(model_->nsite) * 7);
  std::vector<double> geom_poses(static_cast<size_t>(model_->ngeom) * 7);
  for (int i = 0; i < model_->nbody; ++i)
  {
    std::copy(data->xpos + 3 * i, data->xpos + 3 * i + 3, &body_poses[7 * i]);
    std::copy(data->xquat + 4 * i, data->xquat + 4 * i + 4, &body_poses[7 * i + 3]);
  }
  for (int i = 0; i < model_->nsite; ++i)
  {
    std::copy(data->site_xpos + 3 * i, data->site_xpos + 3 * i + 3, &site_poses[7 * i]);
    matToQuat(data->site_xmat + 9 * i, &site_poses[7 * i + 3]);
  }
  for (int i = 0; i < model_->ngeom; ++i)
  {
    std::copy(data->geom_xpos + 3 * i, data->geom_xpos + 3 * i + 3, &geom_poses[7 * i]);
    matToQuat(data->geom_xmat + 9 * i, &geom_poses[7 * i + 3]);
  }

  sensor_msgs::msg::JointState js;
  js.header.stamp = now;
  js.header.frame_id = frame_id_;
  buildJointState(data, js);

  ActuatorStateArrayMsg as;
  as.header.stamp = now;
  as.header.frame_id = frame_id_;
  buildActuatorState(data, as);

  std::vector<mujoco_god_plugin::msg::Contact> contacts;
  contacts.reserve(static_cast<size_t>(data->ncon));
  for (int index = 0; index < data->ncon; ++index)
  {
    const auto & source = data->contact[index];
    mujoco_god_plugin::msg::Contact contact;
    for (int side = 0; side < 2; ++side)
    {
      const int geom = source.geom[side];
      contact.geom_ids[side] = geom;
      if (geom >= 0 && geom < model_->ngeom)
      {
        const char * name = mj_id2name(model_, mjOBJ_GEOM, geom);
        contact.geom_names[side] = name ? name : "geom#" + std::to_string(geom);
        const int body = model_->geom_bodyid[geom];
        const char * body_name = mj_id2name(model_, mjOBJ_BODY, body);
        contact.body_names[side] = body_name ? body_name : "body#" + std::to_string(body);
      }
    }
    contact.distance = source.dist;
    contact.position.x = source.pos[0]; contact.position.y = source.pos[1]; contact.position.z = source.pos[2];
    contact.normal.x = source.frame[0]; contact.normal.y = source.frame[1]; contact.normal.z = source.frame[2];
    contact.dimension = static_cast<uint8_t>(source.dim);
    mj_contactForce(model_, data, index, contact.wrench_contact_frame.data());
    contacts.push_back(std::move(contact));
  }

  std::vector<double> ctrl(static_cast<size_t>(model_->nu));
  std::vector<double> aforce(static_cast<size_t>(model_->nu));
  for (int i = 0; i < model_->nu; ++i)
  {
    ctrl[i] = data->ctrl[i];
    aforce[i] = data->actuator_force[i];
  }

  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    body_poses_ = std::move(body_poses);
    site_poses_ = std::move(site_poses);
    geom_poses_ = std::move(geom_poses);
    ctrl_values_ = std::move(ctrl);
    actuator_force_ = std::move(aforce);
    joint_state_ = js;
    actuator_state_ = as;
    contacts_ = std::move(contacts);
    snapshot_valid_ = true;
  }

  joint_pub_->publish(js);
  actuator_pub_->publish(as);

  for (const auto& o : objects_)
  {
    geometry_msgs::msg::PoseStamped ps;
    bool ok = false;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      fillObjectPose(o.type, o.id, frame_id_, ps, ok);
    }
    if (ok)
    {
      ps.header.stamp = now;
      o.pub->publish(ps);
    }
  }
}

// ============================================================================
// pre_step(): apply the queued writes to the live mjData
// ============================================================================

void GodPlugin::pre_step(mjData* data)
{
  std::vector<PoseWrite> writes;
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    for (const auto& kv : actuator_once_)
    {
      if (kv.first >= 0 && kv.first < model_->nu)
      {
        data->ctrl[kv.first] = kv.second;
      }
    }
    actuator_once_.clear();
    for (const auto& kv : actuator_override_)
    {
      if (kv.first >= 0 && kv.first < model_->nu)
      {
        data->ctrl[kv.first] = kv.second;
      }
    }
    writes.swap(pose_writes_);
  }

  for (const auto& w : writes)
  {
    const int j = freeJointOfBody(w.body);
    if (j < 0)
    {
      continue;
    }
    const int adr = model_->jnt_qposadr[j];
    const int vadr = model_->jnt_dofadr[j];
    for (int i = 0; i < 3; ++i)
    {
      data->qpos[adr + i] = w.pos[i];
    }
    for (int i = 0; i < 4; ++i)
    {
      data->qpos[adr + 3 + i] = w.quat[i];
    }
    for (int i = 0; i < 6; ++i)
    {
      data->qvel[vadr + i] = 0.0;
    }
  }
}

void GodPlugin::on_reset(mjData* data)
{
  {
    std::lock_guard<std::mutex> wlock(write_mutex_);
    pose_writes_.clear();
    actuator_override_.clear();
    actuator_once_.clear();
    override_dirty_.store(false);
  }
  {
    std::lock_guard<std::mutex> ilock(interp_mutex_);
    for (auto& pi : pose_interps_)
    {
      if (pi.goal && pi.goal->is_active())
      {
        auto res = std::make_shared<SetObjectPose::Result>();
        res->success = false;
        res->message = "Aborted by a world reset.";
        pi.goal->abort(res);
      }
    }
    pose_interps_.clear();
    for (auto& ar : actuator_ramps_)
    {
      if (ar.goal && ar.goal->is_active())
      {
        auto res = std::make_shared<SetActuatorControl::Result>();
        res->success = false;
        res->message = "Aborted by a world reset.";
        ar.goal->abort(res);
      }
    }
    actuator_ramps_.clear();
  }
  // Reset services may immediately query poses/joints, before the next periodic
  // update. Publish a fresh snapshot now instead of returning the pre-reset scene.
  last_publish_valid_ = false;
  update(model_, data);
  RCLCPP_INFO(logger_, "GodPlugin state reset.");
}

void GodPlugin::cleanup()
{
  RCLCPP_INFO(logger_, "GodPlugin cleanup.");
  joint_pub_.reset();
  actuator_pub_.reset();
  for (auto& o : objects_)
  {
    o.pub.reset();
  }
  objects_.clear();
  get_pose_srv_.reset();
  set_pose_srv_.reset();
  get_joints_srv_.reset();
  get_contacts_srv_.reset();
  get_actuators_srv_.reset();
  set_actuators_srv_.reset();
  clear_actuators_srv_.reset();
  set_pose_action_.reset();
  set_actuator_action_.reset();
  node_.reset();
}

}  // namespace mujoco_god_plugin

PLUGINLIB_EXPORT_CLASS(mujoco_god_plugin::GodPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
