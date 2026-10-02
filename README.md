# STANCE
Stiffness and Torque Adaptation for Non-linear Compliant Environments

Most passive lower-limb prostheses use a fixed-stiffness ankle where a prosthetist sets its compliance once,
and it stays the same regardless of terrain. This works fine on flat, rigid, predictable ground, where that single
setting can be tuned in advance.

Powered prostheses improve on this by adjusting for walking speed and different rigid terrains. But no existing
control strategy adjusts for compliant ground, where the surface itself gives way unpredictably underfoot,
because the ankle can't be pre-set for mechanics it won't know until the moment it lands. As a consequence,
more than half of lower-limb prosthesis users fall at least once a year, many during "weight acceptance": the
100–200 ms window after heel strike when full body weight shifts onto the prosthetic leg. In that window, the
ankle must get its stiffness right immediately. Too stiff, and it slams the impact into the residual limb, causing
pain. Too compliant, and the leg simply folds.

Which failure occurs isn't up to the user. It's decided by the ground. Concrete, wet grass, gravel, and sand
compress, damp, and grip differently, and some stay dented afterward, so a stiffness that's safe on one surface
can fail on the next. No commercial sensor can measure ground compliance before contact, and vision doesn't
help either. We aim to train a single agent that controls a powered ankle by setting three things at once: target
angle, stiffness, and damping. Moreover, we’ll use only real prosthesis signals: encoder, motor current, load
cells, and IMU.
