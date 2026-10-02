# `model/leg.xml` — the robot

**Owner: A (physics)** · MJCF, MuJoCo's model format. Describes *what exists*, not what it does.

## The chain

MJCF is a tree. Each body hangs off its parent:

```
world
└── foot          3 joints: slide x, slide z, hinge pitch   ← floating base
    └── shank     1 joint:  ankle_hinge                     ← THE MOTOR
        └── upper 1 joint:  pylon_slide                     ← passive spring
            └── body_mass   no joint (rigidly attached)
```

This looks upside down — the foot is at the top of the tree. That is because
the foot is the **floating base**: the one body with no parent. Everything else
hangs off it. Nothing anchors the leg to the world, which is exactly why it can
fall over.

## Your five degrees of freedom

| Joint | Type | Actuated? |
|---|---|---|
| `foot_x` | slide | no |
| `foot_z` | slide | no |
| `foot_pitch` | hinge | no |
| `ankle_hinge` | hinge | **YES** |
| `pylon_slide` | slide | no (passive spring) |

Five joints, one actuator. That is the underactuation claim in the proposal,
and you can verify it in one line:

```python
m = mujoco.MjModel.from_xml_path("model/leg.xml")
print(m.nv, m.nu)   # -> 5 1
```

## Two things that surprise people

**`contype="0" conaffinity="0"` means nothing collides with anything.**
This is what "MuJoCo's contact solver is disabled" means in practice — one
attribute in the `<default>` block, not a code hook or a callback. The floor
geom is purely decorative; the environment moves it at render time so you can
see the dent.

To fall back on MuJoCo's native contact (the week-1/2 safety net), set both to
`1` on `foot_geom` and `floor`, and stop applying your own forces.

**The pylon spring lives in the XML, not in Python.**
`stiffness="120000" damping="800"` on `pylon_slide`. MuJoCo applies it every
step automatically. This is also your difficulty dial: raise the stiffness and
the leg is easier to control, lower it and there is more squash to manage.

## Numbers you may want to change

| What | Where | Current |
|---|---|---|
| Foot length | `foot_geom` size + site positions | 0.26 m |
| Heel / toe position | `heel_site`, `toe_site` pos | −0.07 m / +0.19 m |
| Ankle range | `ankle_hinge` range | ±0.349 rad (±20°) |
| Torque limit | `ankle_motor` ctrlrange | ±150 N·m |
| Body mass | `body_geom` mass | 70 kg |
| Leg length | shank 0.45 + upper 0.40 | 0.85 m |

If you move `heel_site` or `toe_site`, you **must** update `HEEL_OFFSET` and
`TOE_OFFSET` in `ankle_env.py` — the reset geometry maths depends on them.

## Checking your changes

```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('model/leg.xml')
print('DOF:', m.nv, 'actuators:', m.nu)
print('total mass: %.2f kg' % m.body_subtreemass[1])
"
pytest tests/ -q
```
