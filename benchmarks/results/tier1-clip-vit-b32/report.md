# Benchmark: tier1 · clip-vit-b32

2026-09-25 · sceneid 2.0.0.dev0 · commit `7b1f4c6` · manifest sha256 `90e833cf40d1` · embeddings made on Tesla T4

**Setup.** 13 indexed films (9.4 h, 67,523 vectors at 2 fps, 138 MB index) and 5 held-out films. 6,230 queries: 623 base clips × 10 distortions; 4,290 known and 1,940 unknown. 0 failed to render.

Tuned thresholds were chosen on the val split for FAR ≤ 1.0%. Every number below is the **test split** (2,130 known, 970 unknown clips). Brackets are 95% bootstrap intervals over base clips.

## Headline

|  | v1-shipped | v1-tuned | v2-tuned |
|---|---|---|---|
| Identified correctly (DIR) | 74.2% [71.0, 77.2] | 68.4% [65.3, 71.8] | 77.4% [74.8, 80.1] |
| False accepts on unknown clips (FAR) | 5.7% [3.2, 8.4] | 2.3% [0.9, 4.0] | 2.2% [0.3, 4.5] |
| Known clips rejected (FRR) | 25.8% | 31.6% | 22.6% |
| Accepted as the wrong film | 0.0% | 0.0% | 0.0% |
| Best candidate right, before the gate (top-1) | 99.0% [98.4, 99.6] | 99.0% [98.4, 99.6] | 99.2% [98.6, 99.7] |
| Timestamp error, median | 0.19 s | 0.18 s | 0.08 s |
| Timestamp within 1 s | 89.2% [87.3, 91.1] | 89.4% [87.4, 91.4] | 94.5% [93.2, 95.8] |
| Thresholds | min_conf=0.83, min_vote_ratio=0.9 | min_conf=0.925, min_vote_ratio=0.9 | min_score=0.785, max_ratio=0.8 |

## By distortion

DIR on known clips / FAR on unknown clips, per distortion.

| Distortion | v1-shipped DIR / FAR | v1-tuned DIR / FAR | v2-tuned DIR / FAR |
|---|---|---|---|
| original | 89.2% / 9.3% | 89.2% / 3.1% | 99.1% / 3.1% |
| compression | 77.5% / 4.1% | 68.5% / 0.0% | 72.3% / 1.0% |
| downscale | 84.0% / 5.1% | 81.7% / 3.1% | 92.5% / 3.1% |
| crop | 79.3% / 3.1% | 62.4% / 0.0% | 45.1% / 1.0% |
| letterbox | 72.8% / 6.2% | 61.5% / 4.1% | 61.0% / 4.1% |
| mirror | 88.7% / 7.2% | 87.3% / 3.1% | 94.4% / 4.1% |
| color | 82.6% / 8.2% | 79.8% / 3.1% | 88.3% / 1.0% |
| overlay | 6.6% / 0.0% | 3.3% / 0.0% | 62.4% / 0.0% |
| speed | 91.5% / 8.2% | 91.5% / 3.1% | 95.3% / 2.1% |
| screen_recording | 69.5% / 5.1% | 58.7% / 3.1% | 63.8% / 2.1% |

## By clip length

| Length | v1-shipped DIR / FAR | v1-tuned DIR / FAR | v2-tuned DIR / FAR |
|---|---|---|---|
| 3 s | 72.0% / 8.4% | 67.5% / 3.7% | 78.5% / 5.5% |
| 5 s | 76.0% / 4.2% | 69.7% / 0.8% | 79.5% / 0.0% |
| 10 s | 74.8% / 3.7% | 68.2% / 1.7% | 74.2% / 0.0% |

## Unknown clips: false accepts by look-alike group

| Group | v1-shipped FAR | v1-tuned FAR | v2-tuned FAR |
|---|---|---|---|
| blender | 0.0% | 0.0% | 0.0% |
| bw-horror | 3.0% | 2.0% | 3.3% |
| caminandes | 0.0% | 0.0% | 0.0% |
| noir | 12.0% | 5.3% | 3.0% |
| technicolor | 3.3% | 0.0% | 0.7% |

## Known clips by film

| Film | v1-shipped DIR | v1-tuned DIR | v2-tuned DIR |
|---|---|---|---|
| big-buck-bunny | 66.0% | 52.0% | 84.0% |
| caminandes-1 | 45.0% | 35.0% | 45.0% |
| caminandes-2 | 0.0% | 0.0% | 100.0% |
| cosmos-laundromat | 76.7% | 66.7% | 78.3% |
| detour | 81.0% | 79.0% | 76.0% |
| elephants-dream | 46.0% | 40.0% | 84.0% |
| his-girl-friday | 81.0% | 80.7% | 86.0% |
| little-shop-of-horrors | 80.3% | 78.7% | 84.7% |
| night-of-the-living-dead | 65.7% | 56.7% | 56.3% |
| royal-wedding | 85.0% | 75.3% | 82.7% |
| sintel | 28.6% | 22.9% | 57.1% |
| tears-of-steel | 51.7% | 41.7% | 78.3% |
| the-general | 79.7% | 70.7% | 81.7% |

## Speed and size

| Stage (per query) | p50 | p95 | Measured on |
|---|---|---|---|
| Decode frames | 405 ms | 1650 ms | Tesla T4 |
| Embed frames (batched across clips) | 171 ms | 497 ms | Tesla T4 |
| Search (one query, exact) | 133.0 ms | 311.2 ms | Linux-6.6.122+-x86_64-with-glibc2.39 |
| Decide (v1-shipped) | 0.94 ms | 2.85 ms | Linux-6.6.122+-x86_64-with-glibc2.39 |
| Decide (v1-tuned) | 0.78 ms | 1.72 ms | Linux-6.6.122+-x86_64-with-glibc2.39 |
| Decide (v2-tuned) | 0.75 ms | 1.73 ms | Linux-6.6.122+-x86_64-with-glibc2.39 |

Index: 67,523 vectors × 512 dims = 138 MB (exact inner-product search).
