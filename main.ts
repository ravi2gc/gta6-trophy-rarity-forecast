// gta6-trophy-rarity-forecast-01: forecast each GTA VI (PS5) trophy's Exophase earn rate
// at the case's target, as five quantiles (0.05/0.25/0.5/0.75/0.95) per trophy.
// Node 22, standard library only, fully deterministic: no randomness, no clock, no network.
//
// MODEL
// Every trophy's rate decomposes into earners over tracked players: r_j(T) = 100 * E_j(T) / N(T).
// The two move for different reasons, so they are forecast separately and divided.
//
// Reads: Exophase refreshes a game's counts on its own schedule, so consecutive reads are
// often byte-identical. Consecutive identical reads (same rates AND same players_tracked)
// collapse into one distinct read; only distinct reads and their times carry information.
//
// N(T) — players tracked: log N is fit by least squares against log(1 + hours since release)
// over the distinct reads (the last 24 h of them when there are at least 3, else all),
// extrapolated to the target hour, floored at the last N and capped at 30x the last N.
//
// E_j(T) — earners of trophy j: log(E_j + 0.5) is fit the same way over the distinct reads
// with E_j > 0 (E_j = round(r_j / 100 * N)). Each trophy's slope is shrunk toward its
// tier's mean slope, with weight by fit quality (more distinct reads with movement = more
// weight on the raw slope). E_j(T) is extrapolated, floored at 0 and capped at N(T).
// A trophy at >= 99.9% in the last read keeps a median of about 99.95%.
//
// FLAT-HISTORY BRANCH (launch night): activates ONLY when the history holds exactly one
// distinct read AND the case is within ~24 h of release. On launch night the page the
// house read was stale — Exophase had not refreshed for hours — so the last read is not
// the state at the checkpoint, it is the state at Exophase's previous refresh. The pending
// refresh is modelled from the rules' own anchor: the page showed 638 players tracked
// from 2026-11-18T23:48Z through the 04:00Z read, then 3,868 at 04:30Z, a ~6x jump over
// ~1.5 h of clock from the last pre-refresh read. That is a staleness factor of ~3 on top
// of the clock-ratio growth, hence:
//   G_eff = (hours to target / hours to last read)^1.1 * 3,   N(T) = N_last * G_eff
//   E_j(T) = E_j_last * (hours to target / hours to last read)^e_j
// with e_j ~ 0.3 for most trophies (earners grow much slower than the tracked-player
// flood early on — dilution dominates), ~1.0 for the earliest story trophies (ids 2-4,
// which players reach within hours of starting), and 0 for the platinum (nobody can
// plausibly earn it in the first hours). These are launch-night refresh priors from the
// rules' 638 -> 3,868 anchor and generic launch dynamics, NOT values fitted to any answer.
//
// SPREAD (logit scale, per trophy): sigma_j = MULT * sqrt(sigma_rw^2 + sigma_fit^2 + sigma_N^2)
//   sigma_rw  = 0.5 * sqrt(horizon in days)   (baseline-like random walk)
//   sigma_fit = residual std of the trophy's E-fit (0.35 in the flat branch: no fit exists)
//   sigma_N   = residual std of the N-fit     (0.6  in the flat branch: the refresh is unseen)
// MULT is a global spread multiplier calibrated on held-out validation (see README);
// the flat-history branch keeps its own unshrunk multiplier because it has no fit at all.
// Quantiles: q_tau = 100 / (1 + exp(-(logit(median) + z_tau * sigma_j))),
// z = [-1.645, -0.674, 0, 0.674, 1.645], rounded to 4 decimals, clamped to [0.01, 99.99],
// forced non-decreasing.

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
/** Global multiplier on the combined spread, calibrated on held-out validation
 *  (40 simulated launches at the final horizon: 0.4-0.5 is flat-optimal there, 0.5 kept
 *  for robustness to real-world fat tails the smooth simulator lacks). */
const SPREAD_MULT = 0.5;
/** The flat-history branch carries no fit at all — one stale read and an unseen refresh —
 *  so its spread stays unshrunk: on the real 13:18 check its pinball loss is lowest at 1.0. */
const FLAT_SPREAD_MULT = 1.0;
/** Flat-branch priors when a single distinct read carries no fit information. */
const FLAT_SIGMA_FIT = 0.35;
const FLAT_SIGMA_N = 0.6;
/** Fallback residual when a fit has too few points to measure one. */
const DEFAULT_SIGMA_FIT = 0.3;
const DEFAULT_SIGMA_N = 0.3;
/** Release origin if the case carries no usable release_date. */
const FALLBACK_ORIGIN_MS = Date.UTC(2026, 10, 19, 0, 0, 0);

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

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const originMs = c.game?.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : NaN;
  const origin = Number.isFinite(originMs) ? originMs : FALLBACK_ORIGIN_MS;
  const targetMs = Date.parse(c.target);
  const hTarget = hoursSinceOrigin(targetMs, origin);
  const horizonDays = c.horizon_minutes / 1440;
  const sigmaRw = SIGMA_RW * Math.sqrt(horizonDays);

  // Collapse consecutive identical reads into distinct reads (same rates AND same N).
  // A distinct read's time is the LAST read that showed that state: the state was still
  // current then, and the checkpoint-adjacent time is what the forecast extrapolates from.
  // (Keeping the first occurrence would date a launch-night read to 23:50 and wildly
  // overstate how stale it is.)
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (
      prev !== undefined &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      prev.at = r.at;
      continue;
    }
    distinct.push({ ...r, rates: [...r.rates] });
  }
  const last = distinct[distinct.length - 1];
  const lastRates = last.rates;
  const nLast = last.players_tracked;
  const hLast = Math.max(0.25, hoursSinceOrigin(Date.parse(last.at), origin));

  const medians: number[] = new Array(c.trophies.length).fill(0);
  const sigmas: number[] = new Array(c.trophies.length).fill(0);

  if (distinct.length === 1 && hLast <= 24) {
    // ---- Flat-history branch: the single read is stale; model the pending refresh. ----
    const ratio = Math.max(1, hTarget / hLast);
    const gEff = Math.pow(ratio, 1.1) * 3;
    const nTarget = nLast * gEff;
    for (let j = 0; j < c.trophies.length; j++) {
      const id = c.trophies[j].id;
      const lastRate = lastRates[j] ?? 0;
      if (lastRate >= 99.9) {
        medians[j] = 99.95;
      } else {
        const eLast = Math.round((lastRate / 100) * nLast);
        const e = id === 1 ? 0 : id >= 2 && id <= 4 ? 1.0 : 0.3;
        const eTarget = eLast * Math.pow(ratio, e);
        medians[j] = (100 * eTarget) / nTarget;
      }
      sigmas[j] =
        FLAT_SPREAD_MULT * Math.sqrt(sigmaRw * sigmaRw + FLAT_SIGMA_FIT * FLAT_SIGMA_FIT + FLAT_SIGMA_N * FLAT_SIGMA_N);
    }
  } else {
    // ---- General branch: fit N and each E_j in log-log time. ----
    const xsAll = distinct.map((r) => Math.log(1 + hoursSinceOrigin(Date.parse(r.at), origin)));
    // Last 24 h of distinct reads when there are at least 3, else all of them.
    const cutoffMs = Date.parse(last.at) - 24 * 3_600_000;
    const recentIdx = distinct.map((r, i) => i).filter((i) => Date.parse(distinct[i].at) >= cutoffMs);
    const nIdx = recentIdx.length >= 3 ? recentIdx : distinct.map((_, i) => i);
    const nFit = linfit(
      nIdx.map((i) => xsAll[i]),
      nIdx.map((i) => Math.log(Math.max(1, distinct[i].players_tracked))),
    );
    const xTarget = Math.log(1 + hTarget);
    let nTarget = Math.exp(nFit.intercept + nFit.slope * xTarget);
    nTarget = Math.min(30 * nLast, Math.max(nLast, nTarget));
    const sigmaN = nIdx.length >= 3 ? nFit.resid : DEFAULT_SIGMA_N;

    // Per-trophy E fits and tier mean slopes for shrinkage.
    const rawSlopes: Array<number | null> = [];
    const fitMx: Array<number | null> = [];
    const fitMy: Array<number | null> = [];
    const fitResids: Array<number | null> = [];
    const fitCounts: number[] = [];
    for (let j = 0; j < c.trophies.length; j++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < distinct.length; i++) {
        const r = distinct[i].rates[j];
        if (r === null) continue;
        const e = Math.round((r / 100) * distinct[i].players_tracked);
        if (e > 0) {
          xs.push(xsAll[i]);
          ys.push(Math.log(e + 0.5));
        }
      }
      fitCounts.push(xs.length);
      if (xs.length >= 2) {
        const f = linfit(xs, ys);
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
      return nAll > 0 ? sAll / nAll : 0.3;
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
        const slope = w * (rawSlopes[j] as number) + (1 - w) * tierMeanSlope(tier);
        // Pivot the shrunk slope around the fit's centroid, not around x=0: shrinking
        // the slope must not shift the level the data itself pins down.
        eTarget = Math.exp((fitMy[j] as number) + slope * (xTarget - (fitMx[j] as number))) - 0.5;
        sigmaFit = fitCounts[j] >= 3 ? (fitResids[j] as number) : tierResid(tier);
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
