// gta6-trophy-rarity-forecast-01 v2: forecast each GTA VI (PS5) trophy's Exophase earn
// rate at the case's target, as five quantiles (0.05/0.25/0.5/0.75/0.95) per trophy.
// Node 22, standard library only, fully deterministic: no randomness, no clock, no network.
//
// MODEL (v2 — what changed vs v1 is marked [v2] below)
// Every trophy's rate decomposes into earners over tracked players:
//   r_j(T) = 100 * E_j(T) / N(T), forecast separately and divided.
//
// DISTINCT READS [v2]: Exophase refreshes a game's counts on its own schedule, so
// consecutive reads are often byte-identical. Consecutive identical reads (same rates
// AND same players_tracked) collapse into one distinct read. A distinct read's state
// was SET at its FIRST occurrence — the refresh just before the first read that shows
// it — not at its last occurrence: dating a state at its last occurrence pretends a
// state set hours earlier is current, which depresses early points most (a 3 h refresh
// lag is a large fraction of h early on, a small one late) and bends every fit toward
// stronger saturation. So distinct reads are dated at first occurrence here.
// The FIRST distinct read is the pre-release state (638 tracked, launch-night earners):
// it is not a state at its timestamp, so the N-fit drops it (see below).
//
// N(T) — players tracked [v2]: log N is NOT a straight line in log(1+h) over a week:
// the tracked-player flood saturates. v1's power-law extrapolation overshot N(T) by
// ~45% on simulated finals (its medians sat ~28% low on every trophy, uniformly).
// So N is fit with the saturating form
//   N(h) = C * (1+h)^a / (1 + (1+h)^a / S)
// by a deterministic log-space grid search over (a, S) with log C solved by least
// squares, on all distinct reads except the first (pre-release) one. On 80 simulated
// launches the fit recovers a ~ 1.5 and S ~ 1500 and cuts the mean |log N(T) error|
// from 0.45 (power law) to 0.04. When the fit cannot run (too few reads) or finds no
// saturation (S at the grid edge), it falls back to v1's power law on the last 24 h.
// N(T) is floored at the last N and capped at 30x it.
//
// E_j(T) — earners of trophy j: log(E_j + 0.5) is fit against log(1+h) over the
// distinct reads with E_j > 0 and h >= 3 (earner counts are frozen at their
// launch-night values until ~h=3, so earlier reads carry no slope information).
// Each trophy's slope is shrunk toward its tier's mean slope, weighted by fit quality
// (more distinct reads with movement = more weight on the raw slope). (An earliness
// blend was tried for the shrinkage target and dropped: on simulated finals it lost
// to the plain tier mean, and with ~20 distinct reads the raw slopes already carry
// most of the weight.) The shrunk slope pivots around the fit's centroid so
// shrinkage never shifts the level the data pins down. E_j(T) is extrapolated,
// floored at 0 and capped at N(T). A trophy at >= 99.9% in the last read keeps a
// median of about 99.95% (the first trophy: earned within minutes of launching, so
// essentially every tracked player has it — the rules record it staying at 100%
// through the 04:30 refresh while every other rate fell).
//
// EARLINESS PRIOR [v2]: a trophy's launch-night earner count E0_j (among the first
// 638 tracked players) measures how early in a typical playthrough it is reached,
// and early-reached trophies' earners grow fastest as new players start. So the
// prior growth exponent is beta_j = 0.7 * (E0_j / E_med)^0.5, clipped to [0.15, 2.2],
// where E_med is the median launch-night earner count and 0.7 is the centre of the
// earner-growth exponents seen across trophies (the simulator's beta distribution,
// median 0.7, is calibrated to the observed anchors). The platinum gets beta 0.25
// (nobody plausibly earns it in week one). v1's crude buckets (0 / 1.0 for ids 2-4 /
// 0.3 for the rest) underestimated fast-growing early trophies — the worst misses
// in validation — and overestimated some mid-story ones; the smooth prior replaces
// them in the flat branch, where no fit exists to learn growth from data.
//
// FLAT-HISTORY BRANCH (launch night) [v2]: activates ONLY when the history holds
// exactly one distinct read AND the target is within ~48 h of release. The single
// read is the stale pre-release state, so the pending refresh is modelled from the
// rules' own anchor (638 players tracked from 23:48Z through the 04:00Z read, then
// 3,868 at 04:30Z):
//   the stale state is the state at release (h ~ 0): N(h) = 638 * (1+h)^a with
//   a = ln(3868/638) / ln(5.5) ~ 1.057, which reproduces the anchor exactly;
//   the answer shows the state at the last refresh before the target: the first
//   refresh is the anchor one (h = 4.5), later refreshes come ~every 3 h, so the
//   state time is hState = 4.5 for targets up to ~6 h, else hT - 1.5 (the expected
//   age of the last refresh), and 0 if the target precedes the first refresh;
//   beyond the anchor, N continues with exponent 1.5 (the flood steepens past the
//   anchor average as launch day progresses);
//   E_j(hState) = E0_j * (hState/3)^beta_j with beta_j from the earliness prior.
// These are launch-night refresh priors from the rules' anchor and generic launch
// dynamics, NOT values fitted to any answer.
//
// SPREAD (logit scale, per trophy): sigma_j = MULT * sqrt(sigma_rw^2 + sigma_fit^2
// + sigma_N^2), sigma_rw = 0.5 * sqrt(horizon in days) (baseline-like random walk),
// sigma_fit from the E-fit residual, sigma_N from the N-fit residual, each floored
// (0.15 / 0.1) so a suspiciously perfect fit on smooth data cannot claim zero
// uncertainty. In the flat branch there is no fit at all, so both components are
// horizon-dependent priors instead: sigma_fit grows with ln(hState/3) (a beta error
// compounds with the growth horizon) and sigma_N grows past the anchor (the
// continuation exponent is uncertain). MULT is 0.35 in the general branch and 1.0 in
// the flat branch (no fit at all there). Quantiles:
// q_tau = 100 / (1 + exp(-(logit(median) + z_tau * sigma_j))),
// z = [-1.645, -0.674, 0, 0.674, 1.645], rounded to 4 decimals, clamped to
// [0.01, 99.99], forced non-decreasing.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, in logit units per square-root day (as in the house baseline). */
const SIGMA_RW = 0.5;
/** General-branch multiplier on the combined spread. The simulator is smooth (no
 *  day-to-day shocks, no refresh gaps over ~4 h, no model misspecification), so its
 *  own optimum is narrower than this; 0.35 keeps a margin for the real final's
 *  fatter tails while capturing most of the gain over v1's 0.5. */
const SPREAD_MULT = 0.35;
/** The flat-history branch carries no fit at all, so its spread stays unshrunk. */
const FLAT_SPREAD_MULT = 1.0;
/** Flat-branch spread: the earliness prior's uncertainty compounds with the growth
 *  horizon (a beta error of sd ~0.6 moves log E by 0.6 * ln(hState/3)), on top of a
 *  base level uncertainty; the N model's uncertainty grows past the anchor as the
 *  continuation exponent is uncertain. Both are horizon-dependent, not constants. */
const flatSigmaFit = (hState: number): number =>
  0.2 + 0.6 * Math.log(Math.max(1, hState / 3));
const flatSigmaN = (hState: number): number =>
  0.1 + 0.3 * Math.log(Math.max(1, (1 + hState) / (1 + ANCHOR_H_REFRESH)));
/** Floors so a near-perfect fit on smooth data cannot claim zero uncertainty. */
const FLOOR_SIGMA_FIT = 0.15;
const FLOOR_SIGMA_N = 0.1;
/** Fallback residuals when a fit has too few points to measure one. */
const DEFAULT_SIGMA_FIT = 0.3;
const DEFAULT_SIGMA_N = 0.3;
/** Release origin if the case carries no usable release_date. */
const FALLBACK_ORIGIN_MS = Date.UTC(2026, 10, 19, 0, 0, 0);

/** Rules' anchor: the stale launch-night page showed this many players tracked ... */
const ANCHOR_N_STALE = 638;
/** ... and the first refresh, at this hour after release, showed this many. */
const ANCHOR_N_REFRESH = 3868;
const ANCHOR_H_REFRESH = 4.5;
/** N-growth exponent implied by the anchor: ln(3868/638) / ln(1 + 4.5). */
const ANCHOR_A = Math.log(ANCHOR_N_REFRESH / ANCHOR_N_STALE) / Math.log(1 + ANCHOR_H_REFRESH);
/** N-growth exponent past the anchor (the flood steepens past the anchor average as
 *  launch day progresses: the saturating N law's early exponent is ~1.5, recovered
 *  independently by the general branch's saturating fit on simulated finals). */
const ANCHOR_A_LATE = 1.5;
/** Expected age, in hours, of the last refresh before a target well past the first. */
const REFRESH_LAG_H = 1.5;
/** Earliness prior: centre of the earner-growth exponents across trophies. */
const PRIOR_BETA0 = 0.7;
/** Earliness prior: exponent on (E0_j / E_med); 0.5 means the launch-night earner
 *  spread maps weakly onto the growth-exponent spread — the signal is real (the
 *  earliest-reached trophies grow fastest) but noisy (same-count trophies grow very
 *  differently), so the prior differentiates without betting the median on it. */
const PRIOR_BETA_K = 0.5;
const PRIOR_BETA_MIN = 0.15;
const PRIOR_BETA_MAX = 2.2;
/** Platinum growth exponent (nobody plausibly earns it in week one). */
const BETA_PLATINUM = 0.25;
/** Saturating N-fit: grid over a and S (log-spaced), log C solved by least squares. */
const SAT_A_MIN = 0.8;
const SAT_A_MAX = 2.3;
const SAT_A_STEPS = 76;
const SAT_S_MIN = 200;
const SAT_S_MAX = 50000;
const SAT_S_STEPS = 60;
/** Above this S the data show no saturation: fall back to the power law. */
const SAT_S_FALLBACK = 20000;
/** Minimum distinct reads (after dropping the pre-release one) for the saturating fit. */
const SAT_MIN_READS = 6;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Median of numbers (0 when empty). */
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Least squares of y on x. Returns slope, intercept, residual std (0 if < 3 points),
 *  and the centroid (mx, my) so a shrunk slope can pivot around the data, not around x=0. */
function linfit(xs: number[], ys: number[]): { slope: number; intercept: number; resid: number; mx: number; my: number } {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, resid: 0, mx: 0, my: 0 };
  if (n === 1) return { slope: 0, intercept: ys[0], resid: 0, mx: xs[0], my: ys[0] };
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let resid = 0;
  if (n >= 3) {
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const r = ys[i] - (intercept + slope * xs[i]);
      ss += r * r;
    }
    resid = Math.sqrt(ss / (n - 2));
  }
  return { slope, intercept, resid, mx, my };
}

/** Hours from the release origin to an ISO timestamp (never below -0.9, so log(1+h) stays real). */
function hoursSinceOrigin(ms: number, originMs: number): number {
  return Math.max(-0.9, (ms - originMs) / 3_600_000);
}

/**
 * Saturating N-fit: N(h) = C * (1+h)^a / (1 + (1+h)^a / S), fit in log space.
 * Grid-searches (a, S); log C is solved by least squares for each pair. Returns the
 * fitted (a, S, logC) and the log-space residual std, or null when there are too few
 * points or the best fit sits at the no-saturation edge (S >= SAT_S_FALLBACK).
 */
function saturatingNFit(
  hs: number[],
  ns: number[],
): { a: number; s: number; logC: number; resid: number } | null {
  if (hs.length < SAT_MIN_READS) return null;
  const xs = hs.map((h) => Math.log(1 + h));
  const logN = ns.map((n) => Math.log(Math.max(1, n)));
  let best: { ss: number; a: number; s: number; logC: number } | null = null;
  for (let ai = 0; ai < SAT_A_STEPS; ai++) {
    const a = SAT_A_MIN + ((SAT_A_MAX - SAT_A_MIN) * ai) / (SAT_A_STEPS - 1);
    for (let si = 0; si < SAT_S_STEPS; si++) {
      const s = Math.exp(Math.log(SAT_S_MIN) + ((Math.log(SAT_S_MAX) - Math.log(SAT_S_MIN)) * si) / (SAT_S_STEPS - 1));
      // log N_pred = logC + a*x - log(1 + (1+h)^a / S); logC = mean(logN - base).
      let logC = 0;
      const base: number[] = new Array(hs.length);
      for (let i = 0; i < hs.length; i++) {
        base[i] = a * xs[i] - Math.log(1 + Math.pow(1 + hs[i], a) / s);
        logC += logN[i] - base[i];
      }
      logC /= hs.length;
      let ss = 0;
      for (let i = 0; i < hs.length; i++) {
        const r = logN[i] - (logC + base[i]);
        ss += r * r;
      }
      if (best === null || ss < best.ss) best = { ss, a, s, logC };
    }
  }
  if (best === null || best.s >= SAT_S_FALLBACK) return null;
  const resid = hs.length >= 4 ? Math.sqrt(best.ss / (hs.length - 3)) : 0;
  return { a: best.a, s: best.s, logC: best.logC, resid };
}

/** N at hour h under a saturating fit. */
function saturatingNAt(fit: { a: number; s: number; logC: number }, h: number): number {
  const p = Math.pow(1 + h, fit.a);
  return Math.exp(fit.logC) * p / (1 + p / fit.s);
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const originMs = c.game?.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const origin = Number.isFinite(originMs) ? originMs : FALLBACK_ORIGIN_MS;
  const targetMs = Date.parse(c.target);
  const hTarget = hoursSinceOrigin(targetMs, origin);
  const horizonDays = c.horizon_minutes / 1440;
  const sigmaRw = SIGMA_RW * Math.sqrt(horizonDays);

  // Collapse consecutive identical reads into distinct reads (same rates AND same N),
  // dated at their FIRST occurrence: the state was set at the refresh just before it.
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (
      prev !== undefined &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    distinct.push({ ...r, rates: [...r.rates] });
  }
  const last = distinct[distinct.length - 1];
  const lastRates = last.rates;
  const nLast = last.players_tracked;

  // Launch-night earner counts (first distinct read) and their median, for the
  // earliness prior. Id 2's 100% rate (E0 = N) is excluded: it is handled separately.
  const e0: number[] = c.trophies.map((t, j) => {
    const r = distinct[0].rates[j];
    if (r === null || r >= 99.9) return 0;
    return Math.round((r / 100) * distinct[0].players_tracked);
  });
  const eMed = Math.max(1, median(e0.filter((e) => e > 0)));
  const priorBeta = (j: number): number => {
    if (c.trophies[j].id === 1) return BETA_PLATINUM;
    if (e0[j] <= 0) return PRIOR_BETA0;
    const b = PRIOR_BETA0 * Math.pow(e0[j] / eMed, PRIOR_BETA_K);
    return Math.min(PRIOR_BETA_MAX, Math.max(PRIOR_BETA_MIN, b));
  };

  const medians: number[] = new Array(c.trophies.length).fill(0);
  const sigmas: number[] = new Array(c.trophies.length).fill(0);

  if (distinct.length === 1 && hTarget <= 48) {
    // ---- Flat-history branch: the single read is the stale pre-release state. ----
    // The answer shows the state at the last refresh before the target: the first
    // refresh is the rules' anchor one (h = 4.5); later ones come ~every 3 h.
    const hState = hTarget < 4.0 ? 0 : hTarget <= ANCHOR_H_REFRESH ? ANCHOR_H_REFRESH : Math.max(ANCHOR_H_REFRESH, hTarget - REFRESH_LAG_H);
    let nTarget: number;
    if (hState <= ANCHOR_H_REFRESH) {
      nTarget = ANCHOR_N_STALE * Math.pow(1 + hState, ANCHOR_A);
    } else {
      nTarget =
        ANCHOR_N_REFRESH * Math.pow((1 + hState) / (1 + ANCHOR_H_REFRESH), ANCHOR_A_LATE);
    }
    for (let j = 0; j < c.trophies.length; j++) {
      const lastRate = lastRates[j] ?? 0;
      if (lastRate >= 99.9) {
        medians[j] = 99.95;
      } else {
        const eLast = Math.round((lastRate / 100) * nLast);
        const eTarget = hState <= 3 ? eLast : eLast * Math.pow(hState / 3, priorBeta(j));
        medians[j] = (100 * Math.min(eTarget, nTarget)) / nTarget;
      }
      sigmas[j] =
        FLAT_SPREAD_MULT *
        Math.sqrt(sigmaRw * sigmaRw + flatSigmaFit(hState) * flatSigmaFit(hState) + flatSigmaN(hState) * flatSigmaN(hState));
    }
  } else {
    // ---- General branch: fit N and each E_j in log-log time. ----
    const xsAll = distinct.map((r) => Math.log(1 + hoursSinceOrigin(Date.parse(r.at), origin)));
    const hsAll = distinct.map((r) => hoursSinceOrigin(Date.parse(r.at), origin));
    const xTarget = Math.log(1 + hTarget);

    // N(T): saturating fit on all distinct reads except the first (pre-release state),
    // else the power law on the last 24 h of reads (v1's fit).
    let nTarget: number;
    let sigmaN: number;
    const satRows = distinct.slice(1);
    const satFit =
      satRows.length >= SAT_MIN_READS
        ? saturatingNFit(
            satRows.map((r) => hoursSinceOrigin(Date.parse(r.at), origin)),
            satRows.map((r) => r.players_tracked),
          )
        : null;
    if (satFit !== null) {
      nTarget = saturatingNAt(satFit, hTarget);
      sigmaN = Math.max(FLOOR_SIGMA_N, satFit.resid);
    } else {
      const cutoffMs = Date.parse(last.at) - 24 * 3_600_000;
      const recentIdx = distinct.map((r, i) => i).filter((i) => Date.parse(distinct[i].at) >= cutoffMs);
      const nIdx = recentIdx.length >= 3 ? recentIdx : distinct.map((_, i) => i);
      const nFit = linfit(
        nIdx.map((i) => xsAll[i]),
        nIdx.map((i) => Math.log(Math.max(1, distinct[i].players_tracked))),
      );
      nTarget = Math.exp(nFit.intercept + nFit.slope * xTarget);
      sigmaN = nIdx.length >= 3 ? Math.max(FLOOR_SIGMA_N, nFit.resid) : DEFAULT_SIGMA_N;
    }
    nTarget = Math.min(30 * nLast, Math.max(nLast, nTarget));

    // Per-trophy E fits and tier mean slopes for shrinkage.
    const rawSlopes: Array<number | null> = [];
    const fitMx: Array<number | null> = [];
    const fitMy: Array<number | null> = [];
    const fitResids: Array<number | null> = [];
    const fitCounts: number[] = [];
    for (let j = 0; j < c.trophies.length; j++) {
      const xs: number[] = [];
      const ys: number[] = [];
      const xsLate: number[] = [];
      const ysLate: number[] = [];
      for (let i = 0; i < distinct.length; i++) {
        const r = distinct[i].rates[j];
        if (r === null) continue;
        const e = Math.round((r / 100) * distinct[i].players_tracked);
        if (e > 0) {
          xs.push(xsAll[i]);
          ys.push(Math.log(e + 0.5));
          // Earner counts are frozen at launch-night values until ~h=3, so only
          // reads from h>=3 carry slope information; use them when there are >= 2.
          if (hsAll[i] >= 3) {
            xsLate.push(xsAll[i]);
            ysLate.push(Math.log(e + 0.5));
          }
        }
      }
      const useXs = xsLate.length >= 2 ? xsLate : xs;
      const useYs = xsLate.length >= 2 ? ysLate : ys;
      fitCounts.push(useXs.length);
      if (useXs.length >= 2) {
        const f = linfit(useXs, useYs);
        rawSlopes.push(f.slope);
        fitMx.push(f.mx);
        fitMy.push(f.my);
        fitResids.push(f.resid);
      } else {
        rawSlopes.push(null);
        fitMx.push(null);
        fitMy.push(null);
        fitResids.push(null);
      }
    }
    const tierMeanSlope = (tier: string): number => {
      let s = 0;
      let n = 0;
      for (let j = 0; j < c.trophies.length; j++) {
        if (c.trophies[j].tier === tier && rawSlopes[j] !== null) {
          s += rawSlopes[j] as number;
          n++;
        }
      }
      if (n > 0) return s / n;
      let sAll = 0;
      let nAll = 0;
      for (let j = 0; j < c.trophies.length; j++) {
        if (rawSlopes[j] !== null) {
          sAll += rawSlopes[j] as number;
          nAll++;
        }
      }
      return nAll > 0 ? sAll / nAll : PRIOR_BETA0;
    };
    const tierResid = (tier: string): number => {
      let s = 0;
      let n = 0;
      for (let j = 0; j < c.trophies.length; j++) {
        if (c.trophies[j].tier === tier && fitResids[j] !== null && fitCounts[j] >= 3) {
          s += fitResids[j] as number;
          n++;
        }
      }
      return n > 0 ? s / n : DEFAULT_SIGMA_FIT;
    };

    for (let j = 0; j < c.trophies.length; j++) {
      const lastRate = lastRates[j] ?? 0;
      if (lastRate >= 99.9) {
        medians[j] = 99.95;
        sigmas[j] = SPREAD_MULT * Math.sqrt(sigmaRw * sigmaRw + DEFAULT_SIGMA_FIT * DEFAULT_SIGMA_FIT);
        continue;
      }
      const tier = c.trophies[j].tier;
      let eTarget: number;
      let sigmaFit: number;
      if (rawSlopes[j] !== null && fitMx[j] !== null && fitMy[j] !== null) {
        const m = fitCounts[j];
        const w = m / (m + 2); // more distinct reads with movement -> more weight on the raw slope
        // Shrink toward the tier mean slope (ablation: earliness blend removed —
        // on simulated finals it did not beat the tier mean, and the raw slopes
        // already carry most of the weight with ~20 distinct reads).
        const target = tierMeanSlope(tier);
        const slope = w * (rawSlopes[j] as number) + (1 - w) * target;
        // Pivot the shrunk slope around the fit's centroid, not around x=0: shrinking
        // the slope must not shift the level the data itself pins down.
        eTarget = Math.exp((fitMy[j] as number) + slope * (xTarget - (fitMx[j] as number))) - 0.5;
        sigmaFit = fitCounts[j] >= 3 ? Math.max(FLOOR_SIGMA_FIT, fitResids[j] as number) : tierResid(tier);
      } else {
        // Never earned in any distinct read: half an earner, growing like the tier.
        eTarget = 0.5 * Math.pow(Math.max(1, nTarget / nLast), 0.3);
        sigmaFit = tierResid(tier);
      }
      eTarget = Math.min(nTarget, Math.max(0, eTarget));
      medians[j] = (100 * eTarget) / nTarget;
      sigmas[j] = SPREAD_MULT * Math.sqrt(sigmaRw * sigmaRw + sigmaFit * sigmaFit + sigmaN * sigmaN);
    }
  }

  return c.trophies.map((t, j) => {
    const l = logit(medians[j]);
    const qs = Z.map((z) => round4(clamp(100 / (1 + Math.exp(-(l + z * sigmas[j]))))));
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < qs.length; k++) {
      if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    }
    return { id: t.id, q: qs };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
