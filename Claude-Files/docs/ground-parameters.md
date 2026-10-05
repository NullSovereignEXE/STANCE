# Ground parameters: sources, conversions and limitations

Owner: Mukul (`stance_env/ground.py`, `tests/test_ground.py`)

This file records where every number in `GROUND_RANGES` comes from, how each
published value was converted into the model's per-contact-point form, and what
the model does not capture. Every range is either **sourced** (published value
plus a documented conversion), a **proxy** (a related measurement standing in),
or **unsourced** (a deliberate design choice with no data behind it).

> Before citing anything here in a report, check the value against the original
> source. Several were read from secondary tables or abstracts (noted below).

---

## 1. Current ranges

| Parameter | Train | Test | Status |
|---|---|---|---|
| `k0` (N/m) | 5,000 – 100,000 | 2,000 – 5,000 and 100,000 – 200,000 | Partly sourced; deliberately softer than most outdoor ground (§4) |
| `zeta` (–) | 0.05 – 0.30 | 0.30 – 0.60 | Train sourced; test unsourced (§5) |
| `alpha` (–) | 0 – 2 | 2 – 4 | Train sourced; test unsourced (§6) |
| `f_yield` (N) | 300 – 3,000 | 150 – 300 | Sourced for sand (§7) |
| `mu` (–) | 0.30 – 0.90 | 0.15 – 0.30 | Proxy (§8) |

Train and test ranges touch at shared edges but never overlap; this is enforced
by `test_train_and_test_ranges_are_disjoint`.

Damping is sampled as a ratio `zeta` and converted inside `sample()`:
`c = 2 * zeta * sqrt(k0 * M_REF)`.

---

## 2. Model conventions and assumptions

- **Two contact points.** Heel and toe sites (`HEEL_OFFSET = -0.07 m`,
  `TOE_OFFSET = 0.19 m` from the ankle). Each draws its own material.
- **Contact area.** Foot is 0.26 m × 0.08 m = 0.0208 m². Each point is assumed to
  represent half: `A_point = 0.0104 m²` (a 0.08 m × 0.13 m patch). *Assumption.*
- **Mass for damping.** `M_REF = 73.5 kg`, the total mass of `model/leg.xml`,
  assuming one point carries the whole body at heel strike. *Assumption; must
  be updated if the leg model's masses change.*
- **Depth reference.** `D_REF = 0.02 m`.

---

## 3. Conversion rules

Published values come in different forms. Each source type gets its own rule.

| Rule | Input | Conversion | Used for |
|---|---|---|---|
| C1 | Stiffness of a surface under a whole athlete (N/m) | `k_point = K_surface / 2` (two identical springs in parallel) | Nigg sport surfaces |
| C2 | Stiffness under a single test foot (N/m) | Scale by area ratio `A_point / A_test`: linearly for a layer on a firm base, by its square root for deep soil | FIFA turf data |
| C3 | Subgrade modulus `ks` (N/m³, pressure per deflection) | `k_point = ks × A_point` | Sand |
| C4 | Fraction of impact energy returned `r` | `e = sqrt(r)`, `zeta = -ln(e) / sqrt(pi² + ln²(e))` | Damping |
| C5 | Ultimate bearing pressure `q_ult` (Pa) | `f_yield = q_ult × A_point`, with `q_ult = 0.5·γ·B·Nγ·sγ` (Vesić, surface footing, no cohesion) | Sand yield |
| C6 | Bekker sinkage exponent `n` (`p ∝ z^n`) | Least-squares fit of `k0·d·(1 + alpha·d/D_REF)` to `d^n` over 0–2 cm | Stiffening |

Notes on C1: it is exact only while heel and toe are both in contact. During
heel-only contact the modelled surface is softer than the measured one.

Notes on C2: the FIFA Advanced Artificial Athlete uses a 70 mm diameter test foot
(38.5 cm²), so `A_point / A_test = 2.70` (linear) or `1.64` (square root).

---

## 4. Stiffness `k0`

| Surface | Source value | Rule | Per point (N/m) | Status |
|---|---|---|---|---|
| Tumbling floor | 50,000 N/m | C1 | 25,000 | Sourced [1] |
| Gymnastic floor | 120,000 N/m | C1 | 60,000 | Sourced [1] |
| Running track | 240,000 N/m | C1 | 120,000 | Sourced [1] |
| Gymnasium floor | 400,000 N/m | C1 | 200,000 | Sourced [1] |
| Artificial turf, no infill | 79 N/mm | C2 | 130,000 – 214,000 | Sourced [3, 4] |
| Artificial turf, rubber-crumb infill | 201 N/mm | C2 | 330,000 – 543,000 | Sourced [3, 4] |
| Natural turf | 301 N/mm | C2 | 495,000 – 813,000 | Sourced [3, 4] |
| Hybrid turf | 513 N/mm | C2 | 843,000 – 1,386,000 | Sourced [3, 4] |
| Loose sand | 4,800 – 16,000 kN/m³ | C3 | 50,000 – 166,000 | Sourced [7] |
| Medium dense sand | 9,600 – 80,000 kN/m³ | C3 | 100,000 – 832,000 | Sourced [7] |
| Dense sand | 64,000 – 128,000 kN/m³ | C3 | 666,000 – 1,331,000 | Sourced [7] |
| Rubber matting | — | — | — | Proxy: tumbling floor and no-infill turf |
| Gravel | — | — | — | **Unsourced** |

A separate laboratory study gives allowable `ks` of roughly 14 MN/m³ (loose) and
103 MN/m³ (dense) for unsoaked sand, consistent with the table above [8].

**Decision: keep the current band.** The ranges are deliberately softer than
most measured outdoor ground.

- Train (5,000 – 100,000): tumbling and gymnastic floors, the softest loose sand.
- Test, firm (100,000 – 200,000): running track, no-infill turf, upper loose sand.
- Test, soft (2,000 – 5,000): softer than any measured surface. **Unsourced.**
- Above every band: infilled, natural and hybrid turf; medium and dense sand.

The simulated ground therefore covers compliant indoor surfaces and loose sand;
stiffer outdoor ground lies outside the simulated range.

A stability check with random actions (10 episodes per level) showed no
numerical failures up to `k0` = 3,000,000 N/m, though `f_yield` caps the forces,
so this is evidence rather than proof that stiffer bands would be usable.

---

## 5. Damping ratio `zeta`

Ground damping is rarely measured in N·s/m; simulation papers usually choose it
by convention. What is measured is impact energy loss, converted with C4:

| Surface | Energy lost | `zeta` | Source |
|---|---|---|---|
| Best rubber track | 44% | 0.09 | [1] |
| Worst rubber track | 76% | 0.22 | [1] |
| Infilled turf | 85% | 0.29 | [1] |

- **Train 0.05 – 0.30:** covers the measured 0.09 – 0.29 with margin below,
  because drop tests overstate the loss in real running (Nigg cites a model
  estimating only 1–2% loss in a typical running surface) [1].
- **Test 0.30 – 0.60:** heavier, "dead" ground beyond any measurement.
  **Unsourced.**

Sampling `zeta` rather than `c` keeps damping physically consistent with the
stiffness drawn in the same episode.

---

## 6. Stiffening `alpha`

Bekker's law `p ∝ z^n` describes how terrain stiffens with sinkage. Sand:
n = 1.1 in a widely reproduced terrain table [10]; n = 1 to 1.53 in a laboratory
plate study, depending on soil condition and plate shape [11]. Converted with C6
(fit error under 1.5% over 0–2 cm):

| `n` | `alpha` |
|---|---|
| 1.0 | 0.00 |
| 1.1 | 0.18 |
| 1.3 | 0.71 |
| 1.53 | 1.88 |

- **Train 0 – 2:** sand.
- **Test 2 – 4:** moderately beyond measured. **Unsourced.**

The fit depends on the depth window: over 0–4 cm each `alpha` roughly halves.
Artificial turf is reported to stiffen as it compresses [3] but no value was
found. The previous placeholder (train up to 15) implied 16× stiffening at 2 cm,
far beyond the roughly 3× the sand data supports.

---

## 7. Yield force `f_yield`

Sand yield from bearing capacity (C5), with `B = 0.08 m`,
`sγ = 1 - 0.4·B/L = 0.754` for the 0.08 × 0.13 m patch, Vesić's
`Nγ = 2(Nq + 1)·tanφ` [9], friction angles 28–38° typical of sand, and loose
sand failing in punching shear (`tanφ` reduced to 2/3) [9]. Unit weight
`γ = 15–18 kN/m³` is a typical textbook value. *Assumption.*

| Sand | `φ` | Per-point `f_yield` |
|---|---|---|
| Loose (punching shear) | 28–30° (reduced) | 24 – 29 N |
| Medium dense | 32–34° | 156 – 212 N |
| Dense | 36–38° | 318 – 440 N |

Model body weight is 721 N.

- **Train 300 – 3,000:** dense sand at the low end; the upper end represents
  surfaces that essentially do not yield under walking (floors, turf).
- **Test 150 – 300:** medium dense sand.
- **Loose sand is out of scope** (see Limitation 7).

---

## 8. Friction `mu`

Slip-resistance classes quoted from European work: ≥ 0.3 very slip-resistant,
0.2 – 0.29 slip resistant, 0.15 – 0.19 uncertain, < 0.15 slippery [5].

| Condition | Rubber | Ceramic | Source |
|---|---|---|---|
| Water-wetted (bare foot, 800 N) | 0.47 | 0.20 | [5] |
| Detergent-wetted (bare foot, 800 N) | 0.40 | 0.05 | [5] |

Other reference points: 0.4 as the ASTM F3445-24 footwear pass mark [6]; 0.466
for a foot on poppy seeds, a laboratory stand-in for sand [2].

- **Train 0.30 – 0.90:** "very slip-resistant" up to treaded rubber.
- **Test 0.15 – 0.30:** "uncertain" to "slip resistant", including wet ceramic.

Status: **proxy.** The measurements are bare feet on bathroom mats, not a
prosthetic shoe on outdoor ground, and friction fell as load increased. No
friction data was found for gravel or loose soil.

---

## 9. Limitations

1. **No static friction peak.** Granular ground resists more when the foot is
   stationary than when sliding [2]; the `tanh` friction law has one curve.
2. **Equal load split (C1).** Exact only with both points in contact; heel-only
   contact is softer than measured.
3. **No coupling between heel and toe.** Load spreading through plate-like
   surfaces is not modelled; no data was found to calibrate it.
4. **`M_REF` is tied to `leg.xml`.** If the leg masses change, damping drifts.
5. **Damping from drop tests overstates real energy loss** [1]. The C4
   conversion is exact only for a linear spring-damper; stiffening and yield
   make it approximate.
6. **Size effect.** Subgrade moduli come from foundation-scale loading, much
   larger than a foot; the per-point conversion is rough.
7. **Constant yield force.** Real bearing capacity rises with embedment
   depth. With a constant `f_yield`, a foot loaded past it keeps sinking, so
   loose sand (about 25 N per point) cannot be represented.
8. **Stiffening only.** `alpha ≥ 0` cannot represent soils that soften with
   depth (loam, clay: `n < 1`).
9. **Assumed values:** equal contact-area split, sand unit weight, the 0–2 cm
   fitting window for `alpha`.

## 10. Future work

- **Depth-dependent yield:** add the embedment term so yield grows with dent
  depth, enabling loose sand.
- **Heel–toe coupling,** if rollouts show heel-only contact behaving
  unrealistically.
- **Per-cell visual strip** so heel and toe dents render separately.

---

## References

Verify each against the original before citing.

1. D. J. Stefanyshyn and B. M. Nigg, "Energy and Performance Aspects in Sport
   Surfaces," International Sports Surface Science Society (University of
   Calgary). https://www.isss-sportsurfacescience.org/downloads/documents/8P0VK79N7A_Nigg_EnergyandPerform.pdf
2. X. Xiong, A. D. Ames and D. I. Goldman, "A Stability Region Criterion for
   Flat-footed Bipedal Walking on Deformable Granular Terrain," IROS (Georgia
   Tech CRAB Lab). https://crablab.gatech.edu/pages/publications/pdf/xiaobinIROS.pdf
3. James, Thelwell et al., *Sports Engineering* 29:13 (2026),
   doi:10.1007/s12283-026-00546-7. *Full title to be added from the paper.*
4. Advanced Artificial Athlete specification (70 mm test foot), as described in
   *Sports Engineering*, doi:10.1007/s12283-022-00398-x.
5. El-Sherbiny et al., rubber floor mat friction study, *Journal of the Egyptian
   Society of Tribology*. https://jest.journals.ekb.eg/article_80903_6db5f21aef977533ba867b085104b7ae.pdf
   *Author list and title to be confirmed from the paper.*
6. Chavoshian et al., footwear slip resistance against the ASTM F3445-24
   threshold. *Full citation to be added.*
7. J. E. Bowles, *Foundation Analysis and Design*, 5th ed., McGraw-Hill, 1997.
   Subgrade modulus table read via the Strand7 software documentation.
8. "Experimental Evaluation of Size Effects on the Coefficient of Subgrade
   Reaction for Sand under Various Conditions," KFUPM (abstract).
   https://pure.kfupm.edu.sa/en/publications/experimental-evaluation-of-size-effects-on-the-coefficient-of-sub/
9. A. S. Vesić (1973, 1975), bearing-capacity factors and shape factors; local
   and punching shear reduction of `tanφ` to 2/3. Read via secondary teaching
   material; cite the originals.
10. Sand Bekker parameters (n = 1.1) as tabulated in G. S. Rodrigues and
    E. D. R. Lopes, "Analysis of Off-Road Vehicle Performance on Deformable
    Terrain," Proc. SBMAC, citing Bekker and Wong.
11. Sand plate sinkage study (n = 1 to 1.53), *Misr Journal of Agricultural
    Engineering*. https://mjae.journals.ekb.eg/article_102153.html
    *Authors and title to be confirmed.*
