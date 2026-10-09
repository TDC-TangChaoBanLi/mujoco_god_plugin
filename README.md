# MuJoCo God plugin

The plugin exposes generic model pose, joint and actuator inspection/control.

## Contact inspection

`get_contacts` (`mujoco_god_plugin/srv/GetContacts`) returns a read-only contact
snapshot captured on the simulation thread at the configured publication rate.
The response header identifies the snapshot time; service calls do not advance
physics or change the model. Reset refreshes this snapshot.

Each contact includes geometry IDs/names, body names, world position and normal,
signed separation, contact dimension, and MuJoCo contact-frame force/torque.
Negative separation denotes penetration. Expected constrained-interface contacts
must be interpreted by the application; this plugin does not classify contacts
or contain task-specific allowances. Geometry names fall back to `geom#ID` when
unnamed. This sampled API does not establish continuous collision freedom.
