# `ankle_env.py` — what to know and how to defend it

Fifteen items. For each: what it is, why we did it that way, and a one-line
answer if you are asked directly.

Anything not on this list is boilerplate — standard API usage or a readability
choice. For those, explain **what it does**, not why you styled it that way,
and say plainly if it is a convention you followed rather than invented.

---

## 1. `terminated` vs `truncated`

**What it is.** Two separate flags returned by `step()`. `terminated` means the
episode genuinely ended — the leg fell. `truncated` means we cut it off at 100
steps while the leg was still fine.

**Why it matters.** The critic network estimates "how much reward is still to
come". After a fall the answer is zero, because nothing follows. After a time
limit it is *not* zero — the leg would have kept earning +1 per step. If you
report a time limit as `terminated`, the critic learns that surviving is
worthless. Training still runs. The numbers are quietly wrong.

**If asked:** *"Falling is terminal, running out of simulation time is not. They
have to be distinguished or the value function mis-estimates the return at the
end of every successful episode."*

> Note: `truncated = (step_count >= MAX_STEPS) and not terminated`, or both can
> fire on the same step when the leg topples at exactly step 100.

---

## 2. The 11 sensors — and what is deliberately missing

**What it is.** Ankle angle and rate, ankle torque, vertical ground force,
pylon compression, shank tilt and tilt rate, vertical acceleration, previous
action (3).

**Why these.** Every one exists on a real powered prosthesis: joint encoder,
motor current, pylon load cell, IMU. Nothing in the observation requires a
sensor that would not survive field use.

**The important part — what is NOT there.** Ground stiffness, damping and
friction are never given to the agent. Inferring them from the force response
is the entire point of the project. Including them would make the task trivial
and destroy the research claim.

**If asked:** *"The observation is restricted to signals a deployed prosthesis
actually has. Ground properties are excluded by design — the research question
is whether the policy can infer them from its own loading response."*

---

## 3. Ten stacked timesteps

**What it is.** The network sees the last 10 observations concatenated, not
just the current one. 10 × 11 = 110 numbers.

**Why.** A single snapshot says "ground force is 400 N right now". That reads
identically on concrete and on sand — you just arrived there differently. What
separates them is how fast the force *rose*, and that only exists across time.
50 ms at 200 Hz is enough to capture the weight-acceptance transient.

**Why it is also an experiment.** The history ablation (10 / 3 / 1 stacked
steps) is a direct test of the project's central claim. If the 1-step policy
performs as well, the agent is not inferring anything about the ground and the
claim is false.

**If asked:** *"Ground properties are only visible in the shape of the force
rise over time, so a single frame is insufficient. We test this directly with
the history ablation."*

---

## 4. The action is impedance, not torque

**What it is.** The policy outputs a desired ankle angle, a stiffness K, and a
damping B. Those go into an impedance law that produces the torque:

```
tau = K(theta_d - theta) - B * theta_dot
```

**Why.** This is the core formulation of the project. Read the law as a
mechanical system, not a controller: `K(theta_d - theta)` is Hooke's law — a
spring pulling toward a target angle — and `-B*theta_dot` is a damper. So the
ankle behaves like a spring and damper whose constants we can retune every
5 ms.

The agent is not learning a torque trajectory. It is learning **how compliant
to be, and when**. That is what makes this a research formulation rather than
a tuning exercise, and it is what the clinical baseline (fixed-gain FSM
impedance control) cannot do.

**If asked:** *"Commanding impedance rather than torque means the policy
decides the mechanical character of the joint, which is exactly the quantity a
fixed-gain controller has to choose in advance without knowing the terrain."*

---

## 5. K and B are mapped logarithmically

**What it is.** The policy outputs [-1, 1]; we convert to physical units with a
log map, so action 0.0 gives K = 50 N·m/rad rather than 250.

**Why.** K spans 5 to 500 — a factor of 100. With a linear map, half the action
range covers 250–500, and only the last 2% reaches 5–15. The soft end, where
the interesting compliant behaviour lives, would be nearly unreachable. A log
map gives equal resolution per *factor* rather than per absolute increment.

**If asked:** *"Linear mapping would make soft stiffnesses almost impossible to
command, because they occupy a tiny fraction of the action range."*

---

## 6. The torque limit sits outside the network

**What it is.** `np.clip(..., -150, 150)` is applied after the impedance law,
not learned.

**Why.** Whatever the policy outputs, the maximum force the device can produce
is bounded. This is a hard safety constraint that does not depend on the
network behaving well — which matters for the "is RL safe?" objection, because
it converts an argument into a property of the implementation.

**If asked:** *"The actuator limit is enforced in the environment, so the
commanded force is provably capped regardless of policy output."*

---

## 7. The reset geometry

**What it is.** Each episode starts with the heel exactly touching the ground,
carrying randomised approach velocity.

```
foot_z = x_heel*sin(phi) - z_heel*cos(phi) - (XML body offset)
```

**Why the trig.** Rotating a point `(x, z)` about the y-axis by `phi` moves it
to `(x cos phi + z sin phi, -x sin phi + z cos phi)`. Setting the heel's world
height to zero and solving for the foot's height gives the formula.

**Why the subtraction.** A slide joint's `qpos` is an **offset added to the
body's `pos` in the XML**, not an absolute height. The foot body sits at
z = 0.05 in `leg.xml`. Without subtracting it, the leg starts 50 mm in the air
and free-falls for the first 15 steps of every episode — so "heel strike" was
never a heel strike.

**Sanity check worth quoting:** at `phi = 0` the formula gives 0.05 m, which is
exactly the heel site's z offset.

**If asked:** *"Every episode must begin at the moment of heel strike, since
weight acceptance is the window we are studying. The offset term is because
MuJoCo slide joints are relative to the body's declared position."*

---

## 8. Frame skip, and why the torque is computed inside the loop

**What it is.** Physics steps at 1 ms; the policy acts every 5 steps, giving
200 Hz control. The impedance torque is recomputed on **every physics step**,
not once per control step.

**Why.** The torque depends on current ankle angle and velocity, which change
every millisecond. Computing it once and holding it for 5 ms would make the
ankle far less responsive than intended — you would be simulating a slower
controller than you think.

**Why 200 Hz.** Fast enough to resolve a 100–200 ms weight-acceptance
transient; realistic for embedded prosthesis hardware.

**If asked:** *"The impedance law is a function of instantaneous joint state,
so it has to be evaluated at the physics rate even though the policy only
acts at 200 Hz."*

---

## 9. `qfrc_applied` is cleared after every step

**What it is.** One line: `self.data.qfrc_applied[:] = 0.0` after `mj_step`.

**Why.** Applied forces in MuJoCo are **not** automatically cleared. Forget
this and ground forces accumulate step after step, growing without bound. The
simulation does not crash — it just produces nonsense.

**If asked:** *"Applied forces persist in MuJoCo until overwritten, so they
must be zeroed each step or they accumulate."*

---

## 10. The reward — three terms, and why not more

**What it is.**
- Survival: `+1` per step the leg has not failed
- Impact: `-w1 * max(0, F/mg - lambda)^2`
- Smoothness: `-w2 * ||a_t - a_{t-1}||^2`

**Why each exists.** Each is a real cost of prosthetic stance: falls,
residual-limb loading, actuator feasibility. Nothing else. That is the
justification — we specified the **objective**, not the method. Anything more
would be telling the policy *how* to solve the problem, which is the thing we
do not know.

**Why no separate fall penalty.** Termination forfeits all remaining survival
reward. That *is* the penalty, and it scales automatically with how early the
failure happened — a hand-written penalty could not match that without choosing
a magnitude.

**Why the impact term is one-sided.** Ground force is *necessary*; you cannot
support a body without it. Only excess is harmful. Penalising force in general
would make the agent avoid loading the foot at all.

**Why quadratic.** Slightly over threshold is mild, badly over is severe —
matching how tissue damage scales with peak pressure.

**Why force is divided by body weight.** It makes lambda dimensionless and
interpretable: lambda = 2 means "twice body weight is acceptable". Normal
walking peaks near 1.1–1.2 BW.

**If asked:** *"Three terms, each a real clinical cost. We kept it minimal
because every additional term is a loophole the optimiser can exploit, and
because a short reward means the strategy is discovered rather than prescribed."*

---

## 11. The weight tuning question

**What it is.** `W_IMPACT` against the `+1` survival bonus.

**Why it is the only ratio that matters.** Survival pushes toward **stiff** — a
rigid ankle is the easy way not to fold. Impact pushes toward **soft**. That
tension is the physical problem, not an artefact of our design; a real
prosthetist faces exactly this trade-off.

So tuning the ratio asks a question with physical meaning: *how many steps of
successful support is one body weight of excess impact force worth?*

**What we observed.** In a 30k-step smoke run, episode length climbed from 53.7
to 68.6 while reward **fell** from −53.8 to −73.4. The policy learned to survive
longer and was punished for it — it went stiff, generating roughly five times
the impact of random actions.

**If asked:** *"The two terms conflict by design. We watch episode length and
reward separately, because length rising while reward falls means the weighting
is wrong, not the policy."*

---

## 12. Failure thresholds

**What it is.** Topple if leg tilt exceeds ±15°. Collapse if the body drops
below 70% of standing height.

**Why two modes.** They are mechanically different failures. Toppling is a
balance failure; collapse is a load-bearing failure. Reporting fall rate split
by mode is more informative than a single number.

**Why these values.** They are our choice, not derived. 15° is a judgement call
about how far a prosthesis user could plausibly recover from — we measured that
a fixed action survives about 35% of episodes at this threshold, which leaves
enough reward signal to learn from and enough headroom to improve.

**If asked:** *"These are design parameters we chose and then checked. At 15°,
a fixed-gain action survives roughly a third of episodes, which gives the
policy both signal and room to improve."*

---

## 13. Disjoint train and test ranges

**What it is.** Ground properties are drawn from continuous ranges, re-drawn
every episode. A separate, **non-overlapping** range is held back for
evaluation only, under a different environment id.

**Why.** This is the direct answer to the memorisation critique. Because the
ranges are continuous there is no finite set of surfaces to memorise, and
because the test range is disjoint, a policy that only handles what it has seen
will visibly fail at evaluation.

**Why separate ids rather than a flag.** So you cannot accidentally train on
held-out ground. The gap between the two distributions is the headline result;
if they ever mix, it is worthless.

**If asked:** *"Training and evaluation distributions do not overlap, and we
verify that with a test across 60 seeds. The generalisation gap is reported as
a primary result, not a footnote."*

---

## 14. Observation normalisation

**What it is.** `VecNormalize` wrapping the vectorised environments, keeping a
running mean and variance per channel.

**Why it is essential here.** Our channels span over 100,000×: pylon
compression ranges about 0.02, ground force about 2000. A neural network sums
its inputs with weights that all start around the same size. Without
normalisation the force channel dominates completely and pylon compression is
invisible. The policy would train — slowly — and never learn to use half its
sensors. Nothing would warn you.

**The trap.** The statistics must be saved and reloaded with the model, and
`training` set to `False` at evaluation. Otherwise evaluation uses different
scaling than training and every reported number is wrong.

**If asked:** *"Input channels differ by five orders of magnitude, so without
normalisation the gradient is dominated by one sensor. We save the
normalisation statistics alongside the policy so evaluation uses identical
scaling."*

---

## 15. Seeding through `self.np_random`

**What it is.** `super().reset(seed=seed)` initialises `self.np_random`, and
every random draw in the environment uses it rather than bare `np.random`.

**Why.** Reproducibility. Our final results are reported as mean ± standard
deviation over at least three seeds. If seeding does not actually control the
randomness, those numbers are noise and the comparison means nothing.

**If asked:** *"All randomness flows through the environment's seeded
generator, so a given seed reproduces an identical episode. That is what makes
the multi-seed comparison valid."*

---

# Quick-reference table

| # | Item | One-line justification |
|---|---|---|
| 1 | terminated vs truncated | wrong flag teaches the critic that surviving is worthless |
| 2 | 11 sensors, no ground properties | only real hardware signals; inference is the research question |
| 3 | 10-step history | ground is visible in the force rise over time, not one frame |
| 4 | impedance action | the policy chooses compliance, which fixed gains cannot |
| 5 | log-mapped K, B | linear mapping makes soft stiffness unreachable |
| 6 | torque clamp outside the net | hard safety bound independent of policy behaviour |
| 7 | reset geometry | every episode must start exactly at heel strike |
| 8 | torque inside frame-skip | impedance depends on instantaneous joint state |
| 9 | clear `qfrc_applied` | MuJoCo does not auto-clear; forces would accumulate |
| 10 | three reward terms | three real clinical costs; more terms = more loopholes |
| 11 | W_IMPACT ratio | encodes the real stiff-vs-soft trade-off |
| 12 | ±15° / 70% | our design choice, checked for adequate learning signal |
| 13 | disjoint train/test | direct answer to the memorisation critique |
| 14 | normalisation | channels span 100,000x; half the sensors invisible otherwise |
| 15 | seeded np_random | makes the three-seed comparison valid |

---

# For anything not on this list

Standard API usage or readability choices. The right answer explains **what it
does**, not why you styled it that way:

- *"`mj_name2id` converts a name to an array index. We cache the result because
  it runs a thousand times per simulated second."*
- *"`float32` is what Stable-Baselines3 expects; a mismatch triggers a warning
  and a conversion every step."*
- *"The renderer is created lazily because allocating an OpenGL context is
  expensive and fails on headless machines."*
- *"`body_subtreemass` includes everything below that body in the tree, so the
  foot's value is the whole leg."*

If something is a convention you followed rather than invented, say so. "This
is the standard MuJoCo pattern and here is what it does" is a complete answer.
