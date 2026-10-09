# One MuJoCo contact from the simulator's last published snapshot.
int32[2] geom_ids
string[2] geom_names
string[2] body_names
# Signed geometric distance in metres; negative means penetration.
float64 distance
geometry_msgs/Point position
# First axis of MuJoCo's contact frame, in world coordinates.
geometry_msgs/Vector3 normal
uint8 dimension
# Force then torque, expressed in MuJoCo's contact frame.
float64[6] wrench_contact_frame
