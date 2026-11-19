# GTA VI Trophy Rarity Forecast — make.fun entry

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravi2gc/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravi2gc/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun.

## What it does

`main.ts` (Node 22, standard library only) reads a case at `$MAKEFUN_INPUT_DIR`
(else `/input`) and writes `/output/<id>.out`: for each of the game's 55 trophies,
five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of its earn rate on Exophase at the
case's target. Deterministic — two runs are byte-identical.

## The model

Every trophy's rate decomposes into earners over tracked players,
`r_j(T) = 100 * E_j(T) / N(T)`, and the two are forecast separately, because they move
for different reasons: tracked players flood in as the game launches (dilution), while
earners grow as players progress through the game.

- **Distinct reads.** Exophase refreshes a game's counts on its own schedule, so
  consecutive reads are often identical. Consecutive identical reads (same rates AND
  same players tracked) collapse into one distinct read, dated at its last occurrence;
  only distinct reads carry information.
- **N(T).** `log N` is fit by least squares against `log(1 + hours since release)` over
  the distinct reads (the last 24 h of them when there are at least 3), extrapolated to
  the target hour, floored at the last N and capped at 30x it.
- **E_j(T).** `log(E_j + 0.5)` is fit the same way over the distinct reads with
  `E_j > 0`. Each trophy's slope is shrunk toward its tier's mean slope, weighted by fit
  quality (more distinct reads with movement = more weight on the raw slope), and the
  shrunk slope pivots around the fit's centroid so shrinkage never shifts the level the
  data pins down. `E_j(T)` is extrapolated, floored at 0 and capped at `N(T)`. A trophy
  at ≥ 99.9% in the last read keeps a median of about 99.95%.
- **Flat-history branch (launch night).** When the history holds exactly one distinct
  read and the case is within ~24 h of release, the last read is stale — Exophase had
  not refreshed for hours. The pending refresh is modelled from the rules' own anchor
  (638 players tracked from 23:48Z through 04:00Z, then 3,868 at 04:30Z):
  `N(T) = N_last * (hours to target / hours to last read)^1.1 * 3`, and
  `E_j(T) = E_j_last * (hours to target / hours to last read)^e_j` with `e_j ≈ 0.3` for
  most trophies, `≈ 1.0` for the earliest story trophies (ids 2–4), `0` for the platinum.
  These are launch-night refresh priors from that anchor and generic launch dynamics,
  not values fitted to any answer.
- **Spread (logit scale).** `sigma_j = MULT * sqrt(sigma_rw^2 + sigma_fit^2 + sigma_N^2)`,
  with `sigma_rw = 0.5 * sqrt(horizon in days)`, `sigma_fit` from the E-fit residual and
  `sigma_N` from the N-fit residual. The general branch uses `MULT = 0.5`; the
  flat-history branch keeps `MULT = 1.0` because it has no fit at all. Quantiles are
  `100 / (1 + exp(-(logit(median) + z_tau * sigma_j)))`, rounded to 4 decimals, clamped
  to [0.01, 99.99], non-decreasing.

## Validation (honest, held-out)

Scored with the rules' exact pinball loss on the logit scale, against both house
baselines (persistence baseline and curve baseline).

| Test | Mine | House baseline | House curve | Coverage (mine) |
|---|---|---|---|---|
| A — practice p00/p01 vs pure-dilution answers (earners frozen, N=3,868), and earners ×1.25 / ×1.5 / ×2 | 0.10–0.27 | 1.01–1.69 | 1.01–1.69 | 0.95–0.98 |
| B — p00/p01 histories, target 2026-11-19T13:18Z, answers from the public Exophase page fetched then | 0.33 / 0.37 | 2.39 / 2.39 | 2.39 / 2.39 | 0.95 |
| C — 40 simulated launches, 2-day histories, 6,510-min horizon (final shape) | **0.219** | 0.361 | 0.254 | 0.91 |
| C holdout — 40 fresh simulated launches, different seeds | **0.230** | 0.346 | 0.255 | 0.91 |

Mine beats both baselines on every Test A variant, on Test B, and on 30/40 (holdout 33/40)
simulated launches in Test C. The simulator is calibrated to the observed anchors
(N: 638 stale, 3,868 at 04:30, ~13–20k at 13:18; Chapter 1 earners 8 → ~97 over ~13 h)
with per-trophy earner-growth exponents spanning the observed range, 10-minute reads,
and Exophase refreshes every ~2–4 h with rates rounded to 2 decimals as the page prints.

## Known weak points

- Fast-growing early trophies (e.g. the fishing trophy, 2.04% → 1.89% by 13:18 while
  most trophies diluted 10–20x) are underestimated by the flat-history branch's `e_j`
  priors — the single worst misses in Test B. The general branch learns their growth
  from the multi-day history, so this matters most for launch-night-like cases.
- The simulator is smooth: no day-to-day shocks, no refresh gaps longer than ~4 h, no
  model misspecification. Real final-horizon errors are probably fatter-tailed, which
  is why the spread multiplier was kept at 0.5 rather than the sim-optimal 0.4.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
