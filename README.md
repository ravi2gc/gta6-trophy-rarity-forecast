# GTA VI Trophy Rarity Forecast v2 — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravi2gc/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravi2gc/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun. Version 2 of the v1 "Earners over Tracked" entry (dev grade rank 1,300 of 5,376, loss 0.202).

## What it does

`main.ts` (Node 22, standard library only) reads a case at `$MAKEFUN_INPUT_DIR`
(else `/input`) and writes `/output/<id>.out`: for each of the game's 55 trophies,
five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of its earn rate on Exophase at the
case's target. Deterministic — two runs are byte-identical (~40 ms per case).

## The model

Every trophy's rate decomposes into earners over tracked players,
`r_j(T) = 100 * E_j(T) / N(T)`, and the two are forecast separately, because they move
for different reasons: tracked players flood in as the game launches (dilution), while
earners grow as players progress through the game.

- **Distinct reads, dated at first occurrence.** Exophase refreshes a game's counts on
  its own schedule, so consecutive reads are often identical. Consecutive identical
  reads (same rates AND same players tracked) collapse into one distinct read. A
  distinct read's state was *set* at its first occurrence — the refresh just before
  the first read that shows it — not at its last occurrence: dating a state at its
  last occurrence pretends a state set hours earlier is current, which depresses early
  points most (a ~3 h refresh lag is a large fraction of h early on, a small one late)
  and bends every fit toward stronger saturation. The first distinct read is the
  pre-release state (638 tracked, launch-night earners), not a state at its timestamp,
  so the N-fit drops it.
- **N(T) — saturating, not a power law.** `log N` is not a straight line in `log(1+h)`
  over a week: the tracked-player flood saturates. v1's power-law extrapolation
  overshot N(T) by ~45% on simulated finals (its medians sat ~28% low on *every*
  trophy, uniformly). So N is fit with the saturating form
  `N(h) = C * (1+h)^a / (1 + (1+h)^a / S)` by a deterministic log-space grid search
  over (a, S) with log C solved by least squares, on all distinct reads except the
  first (pre-release) one. On 80 simulated launches the fit recovers a ~ 1.5 and
  S ~ 1500 and cuts the mean |log N(T) error| from 0.45 (power law) to 0.04. When the
  fit cannot run (too few reads) or finds no saturation, it falls back to v1's power
  law on the last 24 h. N(T) is floored at the last N and capped at 30x it.
- **E_j(T).** `log(E_j + 0.5)` is fit against `log(1+h)` over the distinct reads with
  `E_j > 0` and h >= 3 (earner counts are frozen at their launch-night values until
  ~h=3, so earlier reads carry no slope information). Each trophy's slope is shrunk
  toward its tier's mean slope, weighted by fit quality (more distinct reads with
  movement = more weight on the raw slope), pivoting around the fit's centroid so
  shrinkage never shifts the level the data pins down. `E_j(T)` is extrapolated,
  floored at 0 and capped at `N(T)`. A trophy at ≥ 99.9% in the last read keeps a
  median of about 99.95% (the first trophy: earned within minutes of launching, so
  essentially every tracked player has it — the rules record it staying at 100%
  through the 04:30 refresh while every other rate fell).
- **Flat-history branch (launch night), re-derived.** When the history holds exactly
  one distinct read and the target is within ~48 h of release, the single read is the
  stale pre-release state. The pending refresh is modelled from the rules' own anchor
  (638 players tracked from 23:48Z through the 04:00Z read, then 3,868 at 04:30Z):
  the stale state is the state at release (h ~ 0), so `N(h) = 638 * (1+h)^a` with
  `a = ln(3868/638) / ln(5.5) ~ 1.057`, which reproduces the anchor exactly (v1's
  `ratio^1.1 * 3` gave 3,667 for p00 and 3,362 for p01 — below the already-observed
  3,868). The answer shows the state at the last refresh before the target: the first
  refresh is the anchor one (h = 4.5), later refreshes come ~every 3 h, so the state
  time is 4.5 for targets up to ~6 h, else the target minus 1.5 h (the expected age
  of the last refresh). Beyond the anchor, N continues with exponent 1.5 (the
  saturating N law's early exponent, recovered independently by the general branch's
  fit on simulated finals).
- **Earners in the flat branch — smooth earliness prior.** A trophy's launch-night
  earner count E0_j (among the first 638 tracked players) measures how early in a
  typical playthrough it is reached, and early-reached trophies' earners grow
  fastest as new players start. So the prior growth exponent is
  `beta_j = 0.7 * (E0_j / E_med)^0.5`, clipped to [0.15, 2.2], where E_med is the
  median launch-night earner count and 0.7 is the centre of the earner-growth
  exponents across trophies (the simulator's beta distribution, median 0.7, is
  calibrated to the observed anchors); the platinum gets beta 0.25. This replaces
  v1's crude buckets (0 / 1.0 for ids 2–4 / 0.3 for the rest), which underestimated
  fast-growing early trophies — the worst misses in validation — and overestimated
  some mid-story ones. The exponent 0.5 is deliberately moderate: the E0 signal is
  real (the earliest-reached trophies grow fastest) but noisy (same-count trophies
  grow very differently), so the prior differentiates without betting the median on
  it. An earliness blend for the general branch's shrinkage target was tried and
  dropped: on simulated finals it lost to the plain tier mean.
- **Spread (logit scale).** `sigma_j = MULT * sqrt(sigma_rw^2 + sigma_fit^2 + sigma_N^2)`,
  with `sigma_rw = 0.5 * sqrt(horizon in days)`, `sigma_fit` from the E-fit residual
  and `sigma_N` from the N-fit residual, each floored (0.15 / 0.1) so a
  suspiciously perfect fit on smooth data cannot claim zero uncertainty. In the flat
  branch there is no fit at all, so both components are horizon-dependent priors
  instead: sigma_fit grows with ln(hState/3) (a beta error compounds with the growth
  horizon) and sigma_N grows past the anchor (the continuation exponent is
  uncertain). MULT is 0.35 in the general branch (the simulator is smooth, so its
  own optimum is narrower than this; 0.35 keeps a margin for the real final's fatter
  tails) and 1.0 in the flat branch. Quantiles are
  `100 / (1 + exp(-(logit(median) + z_tau * sigma_j)))`, rounded to 4 decimals, clamped
  to [0.01, 99.99], non-decreasing.

## Validation (honest, held-out)

Scored with the rules' exact pinball loss on the logit scale, against both house
baselines (persistence baseline and curve baseline). v1 vs v2 on identical harnesses:

| Test | v1 | v2 | House baseline | House curve | Coverage (v2) |
|---|---|---|---|---|---|
| A — practice p00/p01 vs pure-dilution answers (earners frozen, N=3,868), and earners ×1.25 / ×1.5 / ×2 (mean of 8 variants) | 0.204 | **0.168** | 1.01–1.69 | 1.01–1.69 | 0.96–0.98 |
| B — p00/p01 histories, target 2026-11-19T13:18Z, answers from the public Exophase page fetched then | 0.327 / 0.375 | 0.434 / 0.434 | 2.39 / 2.39 | 2.39 / 2.39 | 1.00 |
| C — 40 simulated launches, 2-day histories, 6,510-min horizon (final shape) | 0.219 | **0.146** | 0.361 | 0.254 | 0.92 |
| C holdout — 40 fresh simulated launches, different seeds | 0.230 | **0.150** | 0.346 | 0.255 | 0.92 |

v2 beats v1 clearly on Test A (−18%), Test C (−34%, typical miss 0.17 vs 1.07 points,
beats both baselines on 40/40 launches vs v1's 30/40) and the holdout (−35%, 40/40 vs
33/40). It does **not** beat v1 on Test B (+24%): see below. No parameter was fitted
to the 13:18 live numbers; the flat-branch priors come from the rules' 04:30 anchor,
Test A, and the simulator.

## Known weak points

- **Test B (flat branch at a ~10-hour horizon) is worse than v1 (0.434 vs 0.351).**
  The flat branch's power-law earner growth, `(hState/3)^beta`, over-extrapolates
  typical trophies at long horizons (median 0.159% vs answers mostly 0.06–0.24%) while
  still underestimating the fastest ones (fishing 0.44% vs 1.89%, Chapter 1 0.20% vs
  0.73%): the true growth spread across trophies is wider than the launch-night
  earner counts can resolve, and real earner growth decelerates after launch night
  in a way a power law cannot express. The wide horizon-dependent spread keeps
  coverage at 1.00, but the medians are too compressed. This horizon never occurs in
  grading — the dev grade's targets are 04:30/05:00 (90 minutes, where Test A says
  v2 wins) and the final has a rich history (general branch, where v2 wins big) — but
  it is a real limitation of the flat branch, stated plainly.
- The simulator is smooth: no day-to-day shocks, no refresh gaps longer than ~4 h, no
  model misspecification. Real final-horizon errors are probably fatter-tailed, which
  is why the general-branch spread multiplier was kept at 0.35 rather than the
  sim-optimal ~0.2.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
