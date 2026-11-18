// gta6-twitch-ccu-forecast-01: forecast Twitch concurrent viewers in the
// Grand Theft Auto VI category for hours 2..24 of release day (UTC), as five
// quantiles per hour, from the case's hour-1 print and the comparables'
// past launch curves.
//
// Model (refit from the case file alone on every run, Node stdlib only):
//   Each comparable launch with a complete 24-hour release day contributes
//   one log-ratio curve, lr_h = ln(viewers(h) / viewers(hour 1)) for
//   h = 2..24, at scale x = ln(viewers(hour 1)). The shape of a launch day
//   depends strongly on scale: a huge launch's hour 1 is already inflated
//   by global hype, so its evening peak is only a fraction above hour 1,
//   while a mid-size launch climbs to roughly twice hour 1. So for every
//   hour we fit a weighted least-squares line lr_h = a_h + b_h * x over the
//   comparables and predict at the case's own x, clamped to the
//   comparables' observed scale range (no extrapolation past the biggest
//   launch on record). The fitted median is also winsorised into that
//   hour's observed log-ratio range, so a steep fitted line can never
//   predict a ratio no launch has ever shown.
//   Weights favour launches whose unlock matches the target's rolling
//   midnight-local pattern (1.0) over simultaneous global unlocks (0.15)
//   and unverified ones (0.3). Launches flagged in their research note as
//   early-access-contaminated get weight 0.1 whatever their unlock text
//   says: their release day is really day four or five of play, the
//   early-access peak already spent, so their curve is a flattened,
//   wrong-regime shape for a clean launch like this one (one of them,
//   Hogwarts Legacy, even slipped through the old unlock-text check at
//   full weight because its note mentions midnight local first). Weak
//   launches (free-to-play, slow-burn) keep weight 0.3. Launches whose
//   hour-1 print is under 1,000 viewers are dropped: they were still
//   pre-unlock at hour 1, so their ratios are noise, not signal.
//   The spread per hour is the weighted residual standard deviation of
//   that hour's fit, times 0.40, floored at 0.10 log units. Quantiles are
//   lognormal around the median: q_tau = p1 * exp(mu_h + z_tau * s_h).
//   If the comparables cannot support a fit (too few, degenerate), the
//   program falls back to the house persistence baseline (mu 0,
//   s 0.15 * sqrt(h - 1)) rather than failing the case.
//
// Validation (local scorer implementing the pinball loss of RULES.txt,
// answers = SullyGnome release-day hourly averages, hour-1 values
// verified equal to each case's known print; the scorer reproduces the
// house baseline at 0.387097): mean practice loss 0.139 over p00-p03
// (p00 0.115, p01 0.116, p02 0.118, p03 0.209), vs 0.145 for the previous
// model (p00 0.127, p01 0.135, p02 0.121, p03 0.197). The gain is in the
// three cases the old model already handled; p03 (Spider-Man 2, whose
// midday dip to ~0.37x hour 1 no comparable shows) gets worse, 0.197 to
// 0.209 — the early-access curves were accidentally helping there.
// Leave-one-out CV (spread grid-searched on the other three cases only,
// same procedure for both models): old 0.146, new 0.140, so the gain is
// not just tuning on all four cases at once. The new model is strictly
// better on both the mean practice loss and the LOO-CV loss.
// The 0.40 spread is the modal LOO choice (folds pick 0.40, 0.40, 0.40,
// 0.35) and sits in the flat region of the spread curve; the 0.1
// early-access weight is the LOO argmin over {0, 0.05, 0.1, 0.2}
// (0.141, 0.141, 0.140, 0.143), i.e. a strong downweight, not a hard
// drop, which keeps all 34 curves contributing and is more robust if a
// tier label is ever wrong.
//
// Deterministic: no randomness, no clock, inputs read in name order, one
// fixed output per case. Runs in well under a second on the practice cases.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at the scored levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Milliseconds in one hour. */
const HOUR_MS = 3_600_000;

/** Drop a comparable whose hour-1 print is below this: still pre-unlock. */
const MIN_HOUR_ONE = 1000;
/** Weight of a rolling midnight-local unlock, the target's pattern. */
const WEIGHT_ROLLING = 1;
/** Weight of a simultaneous global unlock. */
const WEIGHT_SIMULTANEOUS = 0.15;
/** Weight of an early-access launch, and of an unverified / other one. */
const WEIGHT_EARLY_ACCESS = 0.3;
const WEIGHT_OTHER = 0.3;
/** Weight of a launch whose research note flags it early-access-contaminated. */
const WEIGHT_CONTAMINATED = 0.1;
/** Weight of a launch flagged weak (free-to-play, slow-burn). */
const WEIGHT_WEAK = 0.3;
/** Residual spread multiplier, and the floor on the lognormal sigma. */
const SPREAD_FACTOR = 0.4;
const SIGMA_FLOOR = 0.1;
/** Fallback spread (the house baseline's) when no fit is possible. */
const FALLBACK_SIGMA = 0.15;
/** Fewest usable comparables a fit needs. */
const MIN_CURVES = 8;

interface KnownRow {
  hour: number;
  viewers: number;
  channels?: number | null;
}

interface HourlyPoint {
  from: string;
  viewers: number;
  channels: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  history_hourly: HourlyPoint[];
  comparables: Comparable[];
}

/** One usable comparable: its scale, its 23 log-ratios, its weight. */
interface LaunchCurve {
  x: number;
  ratios: number[];
  weight: number;
}

/** Round to 2 decimals (half up), as the starter does. */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/**
 * A comparable's weight. The research-note tier wins over the unlock
 * text: an early-access-contaminated launch is downweighted to 0.1 even
 * when its note also mentions a midnight-local unlock, and a weak one to
 * 0.3. Otherwise the weight follows the unlock pattern.
 */
function launchWeight(unlock: string): number {
  const text = (unlock ?? '').toLowerCase();
  const tier = /tier:\s*([a-z-]+)/.exec(text)?.[1] ?? '';
  if (tier.startsWith('early-access')) return WEIGHT_CONTAMINATED;
  if (tier === 'weak') return WEIGHT_WEAK;
  if (text.includes('midnight local') || text.includes('rolling')) return WEIGHT_ROLLING;
  if (text.includes('simultaneous') || text.includes('single global') || text.includes('global unlock')) return WEIGHT_SIMULTANEOUS;
  if (text.includes('early access') || text.includes('early-access')) return WEIGHT_EARLY_ACCESS;
  return WEIGHT_OTHER;
}

/**
 * A comparable's curve, or null when it cannot be used: it needs all 24
 * hourly points of its release day (the GTA VI pre-release comparable has
 * only a handful, on a day that was no launch) and an hour-1 print of at
 * least MIN_HOUR_ONE viewers.
 */
function extractCurve(m: Comparable): LaunchCurve | null {
  const day = Date.parse(m.day_start);
  if (!Number.isFinite(day) || !Array.isArray(m.hourly)) return null;
  const byTime = new Map<number, number>();
  for (const p of m.hourly) {
    const t = Date.parse(p.from);
    if (Number.isFinite(t) && typeof p.viewers === 'number' && p.viewers > 0) byTime.set(t, p.viewers);
  }
  const first = byTime.get(day);
  if (first === undefined || first < MIN_HOUR_ONE) return null;
  const ratios: number[] = [];
  for (let h = 2; h <= 24; h++) {
    const v = byTime.get(day + (h - 1) * HOUR_MS);
    if (v === undefined) return null;
    ratios.push(Math.log(v / first));
  }
  return { x: Math.log(first), ratios, weight: launchWeight(m.unlock) };
}

/** Weighted means, weighted least-squares slope, and residual spread. */
function weightedFit(points: Array<{ x: number; y: number; w: number }>): { a: number; b: number; sd: number } | null {
  let total = 0;
  let meanX = 0;
  let meanY = 0;
  for (const p of points) {
    total += p.w;
    meanX += p.w * p.x;
    meanY += p.w * p.y;
  }
  if (total <= 0) return null;
  meanX /= total;
  meanY /= total;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.x - meanX) * (p.x - meanX);
    sxy += p.w * (p.x - meanX) * (p.y - meanY);
  }
  if (sxx <= 1e-9) return null;
  const b = sxy / sxx;
  const a = meanY - b * meanX;
  let ss = 0;
  for (const p of points) {
    const r = p.y - (a + b * p.x);
    ss += p.w * r * r;
  }
  return { a, b, sd: Math.sqrt(Math.max(ss / total, 1e-9)) };
}

/** One hour's forecast: median log-ratio and lognormal sigma. */
interface HourForecast {
  mu: number;
  sd: number;
}

/**
 * Fit the per-hour median log-ratio curves and spreads for one case.
 * Returns null when the comparables cannot support a fit.
 */
function fitAllHours(c: CaseFile, logP1: number): HourForecast[] | null {
  const curves: LaunchCurve[] = [];
  for (const m of c.comparables ?? []) {
    const curve = extractCurve(m);
    if (curve !== null) curves.push(curve);
  }
  if (curves.length < MIN_CURVES) return null;
  const scales = curves.map((cv) => cv.x);
  // No extrapolation past the observed scale range.
  const x = Math.min(Math.max(...scales), Math.max(Math.min(...scales), logP1));
  const out: HourForecast[] = [];
  for (let h = 0; h < 23; h++) {
    const points = curves.map((cv) => ({ x: cv.x, y: cv.ratios[h] as number, w: cv.weight }));
    const fit = weightedFit(points);
    if (fit === null) return null;
    const observed = points.map((p) => p.y);
    // Winsorise the median into the range some launch has actually shown.
    const mu = Math.min(Math.max(...observed), Math.max(Math.min(...observed), fit.a + fit.b * x));
    out.push({ mu, sd: Math.max(fit.sd * SPREAD_FACTOR, SIGMA_FLOOR) });
  }
  return out;
}

/** The persistence baseline's five quantiles for one hour. */
function baselineQuantiles(p1: number, h: number): number[] {
  return Z.map((z) => round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(h - 1))));
}

/** The five quantiles for one hour from a fitted median and sigma. */
function fittedQuantiles(p1: number, fit: HourForecast): number[] {
  return Z.map((z) => {
    const q = round2(p1 * Math.exp(fit.mu + z * fit.sd));
    return Math.min(1e9, Math.max(1, q));
  });
}

/** A case's hour-1 print, falling back to its last hourly point, then 1. */
function hourOne(c: CaseFile): number {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  if (first !== undefined && typeof first.viewers === 'number' && first.viewers > 0) return Math.max(1, first.viewers);
  const hist = c.history_hourly ?? [];
  for (let i = hist.length - 1; i >= 0; i--) {
    const v = (hist[i] as HourlyPoint).viewers;
    if (typeof v === 'number' && v > 0) return Math.max(1, v);
  }
  return 1;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let hours: Array<{ hour: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    const p1 = hourOne(c);
    const fits = fitAllHours(c, Math.log(p1));
    hours = c.forecast.map(({ hour }) => {
      const fit = fits === null ? null : fits[hour - 2];
      const q = fit === undefined || fit === null ? baselineQuantiles(p1, hour) : fittedQuantiles(p1, fit);
      return { hour, q };
    });
  } catch {
    // A case that cannot be read still gets a valid-shaped persistence
    // forecast rather than no output at all.
    const p1 = 1;
    hours = Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, q: baselineQuantiles(p1, i + 2) }));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
