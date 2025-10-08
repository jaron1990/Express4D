

class BlendShapeIDX_3d_angles:
    bias = 0
    shape           = range(0, 300)
    expression      = range(300 - bias, 400 - bias)
    rotation        = range(400 - bias, 403 - bias)
    jaw_pose        = range(403 - bias, 406 - bias)
    eyes_pose       = range(406 - bias, 412 - bias)
    eyes_pose_l     = range(406 - bias, 409 - bias)
    eyes_pose_r     = range(409 - bias, 412 - bias)
    neck_pose       = range(412 - bias, 415 - bias)
    translation     = range(415 - bias, 418 - bias)

    no_identity_size = translation.stop - expression.start

    FLAME_pose = range(rotation.start, jaw_pose.stop)

# K=4 joints:: neck jaw and eyeballs

class BlendShapeIDX_6d_angles:
    shape           = range(0, 300)
    bias = 300
    expression      = range(300 - bias, 400 - bias)
    rotation        = range(400 - bias, 406 - bias)
    jaw_pose        = range(406 - bias, 412 - bias)
    eyes_pose       = range(412 - bias, 424 - bias)
    eyes_pose_l     = range(412 - bias, 418 - bias)
    eyes_pose_r     = range(418 - bias, 424 - bias)
    neck_pose       = range(424 - bias, 430 - bias)
    translation     = range(430 - bias, 433 - bias)

    FLAME_pose = range(rotation.start, jaw_pose.stop)
