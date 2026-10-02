# P1 — The leg and its motion

**You own:** `model/leg.xml`, `stance_env/viz.py`, `tests/test_leg.py`

---

## `model/leg.xml` — the robot

MJCF is a tree. Each body hangs off its parent:

```
world
└── foot          3 joints: slide x, slide z, hinge pitch   ← floating base
    └── shank     1 joint:  ankle_hinge                     ← THE MOTOR
        └── upper 1 joint:  pylon_slide                     ← passive spring
            └── body_mass   no joint
```

The foot sits at the top of the tree because it is the **floating base** — the
one body with no parent. Nothing anchors the leg to the world, which is exactly
why it can fall over.

**Five joints, one actuator.** That is the underactuation claim in the
proposal, and `test_five_dof_one_motor` verifies it on every run.

### The bug that was already in here

MuJoCo's compiler defaults to **degrees**. The XML wrote
`range="-0.349 0.349"` meaning radians, so the ankle was limited to ±0.349
*degrees* — 57× too small. The ankle could barely move, and nothing would have
told you except a policy that mysteriously refused to learn.

Fixed with `<compiler angle="radian"/>` at the top. Leave it there. If you add
any `euler` or `axisangle` attributes, they are now radians too.

### `contype="0" conaffinity="0"`

Nothing collides with anything. This is what "MuJoCo's contact solver is
disabled" means in practice: one attribute in `<default>`, not a code hook.
The floor geom is decorative — P2's code moves it to show the dent.

To fall back on native contact (the week-1/2 safety net): set both to `1` on
`foot_geom` and `floor`, and stop applying custom forces.

### The pylon spring

`stiffness="120000" damping="800"` on `pylon_slide`. MuJoCo applies it
automatically every step. **This is your difficulty dial** — raise the
stiffness and the leg is easier to control, lower it and there is more squash
to manage. If week 6 goes badly, stiffen it.

### What you may want to change

| What | Where | Current |
|---|---|---|
| Foot length | `foot_geom` size + site positions | 0.26 m |
| Heel / toe | `heel_site`, `toe_site` pos | −0.07 / +0.19 m |
| Ankle range | `ankle_hinge` range | ±0.349 rad |
| Torque limit | `ankle_motor` ctrlrange | ±150 N·m |
| Body mass | `body_geom` mass | 70 kg |
| Materials, lighting, cameras | `<worldbody>` | — |

**If you move a site, tell P3.** `HEEL_OFFSET` and `TOE_OFFSET` in
`ankle_env.py` must match, or the reset geometry places the leg wrongly every
episode. `test_site_offsets_match_constants` catches it.

---

## `stance_env/viz.py` — filming it

```python
rec = Recorder(env.model, track="body_mass")   # or track=None for fixed side view
rec.capture(env.data)                          # after every env.step()
rec.save("videos/rollout.mp4")                 # 20 fps = 5 s of slow motion
```

**Camera tracking is your interesting problem.** The leg translates forward
during stance, so the fixed `side` camera loses it. `track="body_mass"` follows
the body with a free camera, biased downward so the foot stays in shot. Tune
`distance`, `azimuth`, `elevation` and the `lookat` bias until the motion reads
clearly — that is a real visual-design task, not boilerplate.

`side_by_side()` stacks two rollouts horizontally. For the final report, run
the **same seed** through the learned policy and the baseline: on average
ground they look near-identical, on held-out soft ground one stays up and the
other folds. That clip is worth more in a presentation than any plot.

### Headless machines

`stance_env/__init__.py` sets `MUJOCO_GL=egl` when there is no `DISPLAY` on
Linux. It has to happen there, before `mujoco` is imported anywhere — setting
it inside `viz.py` is too late, because `ankle_env` imports mujoco first.

---

## `tests/test_leg.py` — your guard rail

Nine checks. Three earn their place:

- **`test_five_dof_one_motor`** — the proposal's central structural claim
- **`test_site_offsets_match_constants`** — catches XML/Python drift
- **`test_nothing_collides`** — if a geom regains contact, MuJoCo silently adds
  forces on top of P2's, and the physics is quietly wrong

```bash
pytest tests/test_leg.py -v
```

Run before every commit.

---

## Week 1 for you

1. Open `leg.xml`, change a mass or a length, confirm `pytest tests/test_leg.py`
   still passes.
2. `python scripts/record_video.py --track` and watch it. Tune the camera.
3. Add materials and lighting so the leg reads clearly on a projector.
4. Source real numbers: peak ankle torque and range of motion for powered
   prostheses. You have ±20° and 150 N·m — find citations, they go in Report 1.
