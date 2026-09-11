# State of one MuJoCo actuator, combining the commanded control input with the
# resulting joint state.

# Actuator name (as declared in the MJCF).
string name

# Current control input (mjData.ctrl).
float64 ctrl

# Control range of the actuator (0 0 when unbounded).
float64 ctrl_min
float64 ctrl_max

# Actuator force/torque produced in the last step (mjData.actuator_force).
float64 actuator_force

# State of the joint driven by this actuator (NaN when there is none).
float64 joint_position
float64 joint_velocity

# True while the plugin is overriding this actuator's control input.
bool overridden
