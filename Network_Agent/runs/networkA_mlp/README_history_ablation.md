# Network A (MLP) — History Ablation Results

We tested **1, 3, and 10 frames** of sensor history (11 sensors per frame). Each configuration was trained for **300,000 timesteps** with **three seeds (0, 1, 2)**. The tables show the mean across seeds; SD is the sample standard deviation across seeds.

## Training ground

| Metric | F1 (11 inputs) | F3 (33 inputs) | F10 (110 inputs) |
|---|---:|---:|---:|
| Survival (%) | **90.00** | 88.33 | 87.50 |
| Survival SD (percentage points) | 1.00 | 0.76 | 1.00 |
| Mean return | **92.078** | 90.992 | 88.678 |
| Mean toe peak (BW) | **1.655** | 1.710 | 1.698 |
| Mean episode steps | **99.265** | 99.120 | 99.005 |
| Mean target angle (°) | -19.024 | -18.248 | -17.413 |
| Mean stiffness K (Nm/rad) | 450.441 | 472.684 | 447.835 |
| Mean damping B (Nms/rad) | 8.590 | 7.206 | 8.054 |

## Held-out test ground

| Metric | F1 (11 inputs) | F3 (33 inputs) | F10 (110 inputs) |
|---|---:|---:|---:|
| Survival (%) | 45.67 | **50.00** | 42.17 |
| Survival SD (percentage points) | 1.53 | 2.29 | 2.02 |
| Mean return | 77.361 | **78.776** | 75.761 |
| Return SD | 0.484 | 0.199 | 0.353 |
| Mean episode steps | 91.292 | **91.540** | 91.025 |
| Mean target angle (°) | -18.556 | -19.219 | -17.911 |
| Mean stiffness K (Nm/rad) | 418.618 | 416.521 | 414.965 |
| Mean damping B (Nms/rad) | 8.299 | 7.695 | 7.474 |

## Key takeaway

**F1 performed best on the training ground**, while **F3 performed best on the held-out test ground**. F3 improved held-out survival by **7.83 percentage points** over F10 (50.00% vs 42.17%) across the three-seed averages. These are exploratory comparisons based on three training seeds, not proof of statistical significance.

*Source: Network A history-ablation evaluation summaries (F1/F3/F10, seeds 0–2). Test toe peak is omitted because it is capped by the test ground's yield force and is not useful for comparing policies.*
