import { InkPoint, InkStroke, newStrokeId } from "./Stroke";
import { computeBBox } from "./Stroke";

/**
 * Handwriting to shapes (roadmap, 2026-08-27). Hold the pen still at the
 * end of a stroke and the wobbly figure snaps to the clean one you meant:
 * line, triangle, rectangle, circle, ellipse, arrow or star.
 *
 * The dwell IS the request. Recognition never runs on an ordinary lift,
 * so nothing about normal writing changes - a deliberate ~third-of-a-
 * second hold is the whole gesture. Everything here is pure geometry on
 * the stroke's own points: no models, no network, works identically on
 * every platform.
 *
 * The recognizer is honest about ambiguity: it classifies only when the
 * fit error is small relative to the figure's size, and returns null
 * otherwise - a null keeps the freehand stroke exactly as drawn.
 */

/** The hold that asks for a snap: this long ... */
export const DWELL_MS = 350;
/** ... within this world-unit radius (a few screen px). */
export const DWELL_RADIUS = 3;
/** Ignore flicks and dots: a snap candidate has some size to it. */
const MIN_PATH_LENGTH = 24;
/** Endpoints closer than this fraction of path length close the figure. */
const CLOSURE_FRACTION = 0.22;
/** RDP simplification tolerance as a fraction of the bbox diagonal. */
/** Max mean radial error for a circle/ellipse, fraction of mean radius. */
const ROUND_TOLERANCE = 0.16;
/** Max perpendicular deviation for a line, fraction of its length. */
const LINE_TOLERANCE = 0.05;
/** Synthesized point spacing in world units. */
const SYNTH_STEP = 3;

export type SnappedKind = "line" | "triangle" | "rectangle" | "circle" | "ellipse" | "arrow" | "star";

export interface SnapResult {
	kind: SnappedKind;
	points: InkPoint[];
}

interface P {
	x: number;
	y: number;
}

function dist(a: P, b: P): number {
	return Math.hypot(a.x - b.x, a.y - b.y);
}

function pathLength(pts: readonly P[]): number {
	let len = 0;
	for (let i = 1; i < pts.length; i++) len += dist(pts[i - 1]!, pts[i]!);
	return len;
}

/**
 * Did the stroke end with a deliberate hold? True when the final DWELL_MS
 * of samples stay inside DWELL_RADIUS. Returns the index where the dwell
 * begins (so the recognizer can drop the parked tail), or null.
 */
export function dwellStart(points: readonly InkPoint[]): number | null {
	if (points.length < 4) return null;
	// A dot is not a snap request, however long it is held: the WHOLE
	// stroke sitting inside the dwell radius means nothing was drawn.
	let aMinX = Infinity;
	let aMinY = Infinity;
	let aMaxX = -Infinity;
	let aMaxY = -Infinity;
	for (const p of points) {
		if (p.x < aMinX) aMinX = p.x;
		if (p.y < aMinY) aMinY = p.y;
		if (p.x > aMaxX) aMaxX = p.x;
		if (p.y > aMaxY) aMaxY = p.y;
	}
	if (Math.hypot(aMaxX - aMinX, aMaxY - aMinY) <= DWELL_RADIUS * 2) return null;
	const endT = points[points.length - 1]!.t;
	let i = points.length - 1;
	while (i > 0 && endT - points[i - 1]!.t <= DWELL_MS) i--;
	if (i < 4) return null; // all (or nearly all) dwell: that is a dot
	const tail = points.slice(i);
	if ((points[points.length - 1]!.t - tail[0]!.t) < DWELL_MS * 0.8) return null;
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	for (const p of tail) {
		if (p.x < minX) minX = p.x;
		if (p.y < minY) minY = p.y;
		if (p.x > maxX) maxX = p.x;
		if (p.y > maxY) maxY = p.y;
	}
	if (Math.hypot(maxX - minX, maxY - minY) > DWELL_RADIUS * 2) return null;
	return i;
}

/** Ramer-Douglas-Peucker, returning the kept vertices. */
export function simplify(pts: readonly P[], tolerance: number): P[] {
	if (pts.length <= 2) return [...pts];
	const keep = new Array<boolean>(pts.length).fill(false);
	keep[0] = true;
	keep[pts.length - 1] = true;
	const stack: Array<[number, number]> = [[0, pts.length - 1]];
	while (stack.length > 0) {
		const [a, b] = stack.pop()!;
		const pa = pts[a]!;
		const pb = pts[b]!;
		const len = dist(pa, pb);
		let worst = -1;
		let worstD = tolerance;
		for (let i = a + 1; i < b; i++) {
			const p = pts[i]!;
			const d =
				len === 0
					? dist(p, pa)
					: Math.abs((pb.x - pa.x) * (pa.y - p.y) - (pa.x - p.x) * (pb.y - pa.y)) / len;
			if (d > worstD) {
				worstD = d;
				worst = i;
			}
		}
		if (worst >= 0) {
			keep[worst] = true;
			stack.push([a, worst], [worst, b]);
		}
	}
	return pts.filter((_, i) => keep[i]);
}

/** Resample a closed outline to n points at equal arc spacing. */
function resampleClosed(pts: readonly P[], n: number): P[] {
	const closed = [...pts, pts[0]!];
	const total = pathLength(closed);
	const step = total / n;
	const out: P[] = [];
	let i = 0;
	let acc = 0;
	for (let k = 0; k < n; k++) {
		const target = k * step;
		while (i < closed.length - 2 && acc + dist(closed[i]!, closed[i + 1]!) < target) {
			acc += dist(closed[i]!, closed[i + 1]!);
			i++;
		}
		const a = closed[i]!;
		const b = closed[i + 1]!;
		const seg = dist(a, b);
		const f = seg === 0 ? 0 : (target - acc) / seg;
		out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f });
	}
	return out;
}

function meanRadialError(pts: readonly P[], cx: number, cy: number, rx: number, ry: number): number {
	let err = 0;
	for (const p of pts) {
		// Normalized radial distance for an axis-aligned ellipse: 1 on the rim.
		const nx = (p.x - cx) / rx;
		const ny = (p.y - cy) / ry;
		err += Math.abs(Math.hypot(nx, ny) - 1);
	}
	return err / pts.length;
}

function classifyClosed(body: readonly P[], gapFrac: number): SnapResult | null {
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	for (const p of body) {
		if (p.x < minX) minX = p.x;
		if (p.y < minY) minY = p.y;
		if (p.x > maxX) maxX = p.x;
		if (p.y > maxY) maxY = p.y;
	}
	const w = maxX - minX;
	const h = maxY - minY;
	if (w < 4 || h < 4) return null;
	const cx = (minX + maxX) / 2;
	const cy = (minY + maxY) / 2;

	const roundErr = meanRadialError(body, cx, cy, w / 2, h / 2);
	const diag = Math.hypot(w, h);

	// CORNERS OUTRANK ROUNDNESS, found by curvature concentration on an
	// equal-arc resampling of the ring. A corner is a large direction
	// change inside a small window (a square's 90 deg arrives all at
	// once); a circle spreads its 360 evenly (about 34 deg per window at
	// this k), so no window ever spikes. Radial fit alone once called
	// squares circles - a perfect square's mean radial error is 0.148,
	// inside any tolerance loose enough to accept a hand circle
	// (hardware, 2026-08-27).
	const N = 64;
	const K = 3;
	const ring = resampleClosed(body, N);
	// Ring samples past this index lie on the synthetic closing chord, not
	// on ink. A gap that swallowed a whole corner made the chord read as a
	// triangle edge and the classifier fit evidence the pen never drew
	// (hardware, 2026-08-27): chord samples are masked out of corner
	// detection and fit, and junction spikes where chord meets ink too.
	// Only a substantial gap can fake an edge; a small one (seam at a
	// corner) must not mask the real corner sitting there.
	const chordStart = gapFrac > 0.05 ? Math.floor(N * (1 - gapFrac)) : N;
	const onChord = (i: number): boolean => chordStart < N && (i >= chordStart - K || i < K);
	const turns: number[] = new Array<number>(N);
	for (let i = 0; i < N; i++) {
		const a = ring[(i - K + N) % N]!;
		const c = ring[i]!;
		const b = ring[(i + K) % N]!;
		const a1 = Math.atan2(c.y - a.y, c.x - a.x);
		const a2 = Math.atan2(b.y - c.y, b.x - c.x);
		let turn = Math.abs(a2 - a1);
		if (turn > Math.PI) turn = 2 * Math.PI - turn;
		turns[i] = turn;
	}
	// Adaptive threshold: a corner is a turn that towers over the ring's
	// OWN median. A square's median is its near-zero edge turn, so even a
	// well-rounded corner (40-55 deg in the window) towers and counts -
	// the fixed 55 deg bar dropped those and made squares into triangles
	// (hardware, 2026-08-27). A circle's median IS its uniform ~34 deg, so
	// its threshold rises far above anything on the ring.
	const sortedTurns = [...turns].sort((a, b) => a - b);
	const median = sortedTurns[N >> 1]!;
	const threshold = Math.max((40 * Math.PI) / 180, median * 3);
	const strongIdx: number[] = [];
	for (let i = 0; i < N; i++) {
		if (onChord(i)) continue;
		if (turns[i]! <= threshold) continue;
		let isMax = true;
		for (let d = -K - 1; d <= K + 1; d++) {
			if (d === 0) continue;
			const j = (i + d + N) % N;
			if (turns[j]! > turns[i]! || (turns[j]! === turns[i]! && j < i)) {
				isMax = false;
				break;
			}
		}
		if (isMax) strongIdx.push(i);
	}
	let strong: P[] = strongIdx.map((i) => ring[i]!);

	// A five-pointed star drawn point-to-point without lifting the pen has
	// ten corners packed into one ring, tighter than the generic
	// curvature-peak search above (tuned for triangle/rectangle/circle)
	// can reliably resolve - adjacent tip/valley corners fall inside each
	// other's non-max-suppression window and half of them get silently
	// dropped. Its own pass, straight on the radial profile (distance from
	// centroid around the ring), does not depend on that resolution and
	// runs independently rather than gated on `strong`.
	const star = classifyStar(ring, cx, cy, diag, onChord);
	if (star) return star;

	// FIT-VERIFIED synthesis: the shape has to actually hug the drawn
	// outline to win. One rounded corner kept slipping under every
	// threshold and squares came out triangles (hardware, 2026-08-27) -
	// now a triangle that fits poorly gets the strongest suppressed peak
	// rescued as a fourth corner and the two candidates compete on fit,
	// and any polygon that still fits badly falls back to freehand.
	const FIT_TOL = 0.05;
	if (strong.length === 3) {
		let err = polygonFitError(ring, strong, diag, onChord);
		if (err > FIT_TOL) {
			let best = -1;
			let bestTurn = (25 * Math.PI) / 180;
			for (let i = 0; i < N; i++) {
				if (onChord(i)) continue;
				if (strongIdx.some((k) => Math.min(Math.abs(i - k), N - Math.abs(i - k)) <= K + 1)) continue;
				if (turns[i]! > bestTurn) {
					bestTurn = turns[i]!;
					best = i;
				}
			}
			if (best >= 0) {
				const rescuedIdx = [...strongIdx, best].sort((a, b) => a - b);
				const rescued = rescuedIdx.map((i) => ring[i]!);
				const e2 = polygonFitError(ring, rescued, diag, onChord);
				if (e2 < err) {
					strong = rescued;
					err = e2;
				}
			}
		}
		// A gap that swallowed a corner leaves only three drawn corners.
		// The undrawn one sits where the outline's two loose ends point:
		// complete it from the edge directions at the gap and let the quad
		// compete on (masked) fit.
		if (strong.length === 3 && gapFrac > 0.06) {
			const completed = completeGapCorner(body, w, h);
			if (completed) {
				const quad = [...strong, completed];
				const e3 = polygonFitError(ring, quad, diag, onChord);
				if (e3 < err && e3 <= FIT_TOL) {
					strong = quad;
					err = e3;
				}
			}
		}
		if (err > FIT_TOL) return null;
		if (strong.length === 3) return { kind: "triangle", points: synthPolygon(strong) };
	}
	if (strong.length === 4) {
		if (polygonFitError(ring, strong, diag, onChord) > FIT_TOL) return null;
		// Snap near-axis-aligned quads to the true rectangle of their bbox;
		// keep a deliberate diamond/parallelogram on its own corners.
		const axisAligned = strong.every(
			(c) =>
				Math.min(Math.abs(c.x - minX), Math.abs(c.x - maxX)) < w * 0.2 &&
				Math.min(Math.abs(c.y - minY), Math.abs(c.y - maxY)) < h * 0.2
		);
		if (axisAligned) {
			return {
				kind: "rectangle",
				points: synthPolygon([
					{ x: minX, y: minY },
					{ x: maxX, y: minY },
					{ x: maxX, y: maxY },
					{ x: minX, y: maxY },
				]),
			};
		}
		return { kind: "rectangle", points: synthPolygon(strong) };
	}
	// Round needs a cornerless outline (an extreme ellipse's tips can
	// register as up to two curvature spikes); a soft-cornered square
	// rejected above stays freehand rather than becoming a circle.
	if (strong.length <= 2 && roundErr < ROUND_TOLERANCE) {
		const aspect = w / h;
		if (aspect > 0.8 && aspect < 1.25) {
			const r = (w + h) / 4;
			return { kind: "circle", points: synthEllipse(cx, cy, r, r) };
		}
		return { kind: "ellipse", points: synthEllipse(cx, cy, w / 2, h / 2) };
	}
	return null;
}

/**
 * The corner the pen never drew: where the outline's two loose ends point.
 * Averages a few samples at each end for a stable direction; null when the
 * lines are near-parallel or the intersection flies far outside the figure.
 */
/**
 * Five-pointed star, found on the ring's radial profile (distance from the
 * centroid around the outline) rather than the shared curvature-peak
 * search: ten corners packed into one ring sit closer together than that
 * search's non-max-suppression window can reliably separate, dropping
 * half of them. A star's radius simply alternates far/near/far/near five
 * times, which is a smaller, more direct signal to look for directly.
 */
function classifyStar(
	ring: readonly P[],
	cx: number,
	cy: number,
	diag: number,
	onChord: (i: number) => boolean
): SnapResult | null {
	const N = ring.length;
	const r = ring.map((p) => Math.hypot(p.x - cx, p.y - cy));

	// Local extrema of the radial profile, each at least MIN_GAP samples
	// from the last one accepted (a star's ten corners are evenly spaced;
	// noise produces extra tiny wiggles much closer together than that).
	const MIN_GAP = Math.floor(N / 14);
	function extrema(wantMax: boolean): number[] {
		const idx: number[] = [];
		for (let i = 0; i < N; i++) {
			if (onChord(i)) continue;
			const prev = r[(i - 1 + N) % N]!;
			const cur = r[i]!;
			const next = r[(i + 1) % N]!;
			const isPeak = wantMax ? cur >= prev && cur >= next : cur <= prev && cur <= next;
			if (!isPeak) continue;
			if (idx.length > 0 && Math.min(i - idx[idx.length - 1]!, N - (i - idx[idx.length - 1]!)) < MIN_GAP) {
				// Keep whichever of the two nearby candidates is the more
				// extreme point rather than just the first one found.
				const better = wantMax ? cur > r[idx[idx.length - 1]!]! : cur < r[idx[idx.length - 1]!]!;
				if (better) idx[idx.length - 1] = i;
				continue;
			}
			idx.push(i);
		}
		// Wrap-around: the first and last accepted peaks may really be the
		// same one straddling index 0.
		if (idx.length > 1) {
			const first = idx[0]!;
			const last = idx[idx.length - 1]!;
			if (Math.min(first + (N - last), N - (first + (N - last))) < MIN_GAP) idx.pop();
		}
		return idx;
	}

	const maxima = extrema(true);
	const minima = extrema(false);
	if (maxima.length !== 5 || minima.length !== 5) return null;

	// Outer tips and inner valleys must actually alternate around the
	// ring, and the tips must reach well past the valleys - otherwise
	// this is some other ten-cornered wobble, not a star.
	const tagged = [...maxima.map((i) => ({ i, outer: true })), ...minima.map((i) => ({ i, outer: false }))].sort(
		(a, b) => a.i - b.i
	);
	for (let k = 0; k < 10; k++) {
		if (tagged[k]!.outer === tagged[(k + 1) % 10]!.outer) return null;
	}
	const outerAvg = maxima.reduce((s, i) => s + r[i]!, 0) / 5;
	const innerAvg = minima.reduce((s, i) => s + r[i]!, 0) / 5;
	if (outerAvg < innerAvg * 1.3) return null;
	if (outerAvg * 2 < diag * 0.3) return null; // too small to be a confident read

	// Fit check against a regular star at this radius pair, oriented from
	// the first detected tip - same bar the other shapes hold to.
	const startAngle = Math.atan2(ring[maxima[0]!]!.y - cy, ring[maxima[0]!]!.x - cx);
	const idealCorners: P[] = [];
	for (let k = 0; k < 10; k++) {
		const rad = k % 2 === 0 ? outerAvg : innerAvg;
		const angle = startAngle + (k * Math.PI) / 5;
		idealCorners.push({ x: cx + Math.cos(angle) * rad, y: cy + Math.sin(angle) * rad });
	}
	if (polygonFitError(ring, idealCorners, diag, onChord) > 0.12) return null;

	return { kind: "star", points: synthPolygon(idealCorners) };
}

function completeGapCorner(body: readonly P[], w: number, h: number): P | null {
	const m = Math.min(6, Math.floor(body.length / 4));
	if (m < 2) return null;
	const a0 = body[0]!;
	const a1 = body[m]!;
	const b0 = body[body.length - 1]!;
	const b1 = body[body.length - 1 - m]!;
	// Rays leaving the drawn ink at each end: from a1 through a0, from b1
	// through b0.
	const dax = a0.x - a1.x;
	const day = a0.y - a1.y;
	const dbx = b0.x - b1.x;
	const dby = b0.y - b1.y;
	const denom = dax * dby - day * dbx;
	if (Math.abs(denom) < 1e-6) return null;
	const t = ((b0.x - a0.x) * dby - (b0.y - a0.y) * dbx) / denom;
	const x = a0.x + dax * t;
	const y = a0.y + day * t;
	const cx = (Math.min(a0.x, b0.x) + Math.max(a0.x, b0.x)) / 2;
	const cy = (Math.min(a0.y, b0.y) + Math.max(a0.y, b0.y)) / 2;
	if (Math.hypot(x - cx, y - cy) > Math.hypot(w, h)) return null;
	return { x, y };
}

/** Mean distance from ring points to the polygon outline, over diag. */
function polygonFitError(
	ring: readonly P[],
	corners: readonly P[],
	diag: number,
	skip?: (i: number) => boolean
): number {
	let total = 0;
	let counted = 0;
	for (let ri = 0; ri < ring.length; ri++) {
		if (skip?.(ri)) continue;
		const p = ring[ri]!;
		counted++;
		let best = Infinity;
		for (let i = 0; i < corners.length; i++) {
			const a = corners[i]!;
			const b = corners[(i + 1) % corners.length]!;
			const vx = b.x - a.x;
			const vy = b.y - a.y;
			const ll = vx * vx + vy * vy;
			const t = ll === 0 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / ll));
			const d = Math.hypot(p.x - (a.x + vx * t), p.y - (a.y + vy * t));
			if (d < best) best = d;
		}
		total += best;
	}
	if (counted === 0) return Infinity;
	return total / counted / diag;
}

/**
 * Synthesize an arrow: curved shaft from the body points, clean arrowhead at tip.
 * HEAD_RATIO: arrowhead arm length relative to shaft length.
 * HEAD_ANGLE: half-opening angle of the arrowhead (radians).
 * HEAD_MAX_PX: absolute cap on the arrowhead arm so oversized drawn wings
 *   do not produce an unreadably huge synthesized head.
 */
const ARROW_HEAD_RATIO = 0.18;
const ARROW_HEAD_ANGLE = (Math.PI * 25) / 180; // 25 deg
const ARROW_HEAD_MAX_PX = 60; // world-unit ceiling for the arm length

/**
 * Synthesize an arrow that keeps the drawn shaft curve but replaces the
 * arrowhead with a clean, proportional one.
 *
 * shaftBody: the stroke points from tail up to (but not past) where the
 *   arrowhead wings begin. The last point in this array is near the tip.
 * tip: the geometrically farthest point from the tail (the apex).
 * shaftLen: straight-line distance tail-to-tip, used to size the arrowhead.
 */
function synthArrow(shaftBody: readonly P[], tip: P, shaftLen: number): InkPoint[] {
	// The synthesized head must be symmetric around the same axis used to
	// recognize the arrow. Using the last few hand-drawn samples made the
	// two wings skew whenever the approach was curved or noisy.
	const tail = shaftBody[0]!;
	const angle = Math.atan2(tip.y - tail.y, tip.x - tail.x);

	// Cap the arrowhead arm: proportional to shaft but never huge.
	const headLen = Math.min(shaftLen * ARROW_HEAD_RATIO, ARROW_HEAD_MAX_PX);
	const leftWing: P = {
		x: tip.x - headLen * Math.cos(angle - ARROW_HEAD_ANGLE),
		y: tip.y - headLen * Math.sin(angle - ARROW_HEAD_ANGLE),
	};
	const rightWing: P = {
		x: tip.x - headLen * Math.cos(angle + ARROW_HEAD_ANGLE),
		y: tip.y - headLen * Math.sin(angle + ARROW_HEAD_ANGLE),
	};

	// The snapped arrow uses the same straight-line synthesis as a snapped
	// line. The user's hand-drawn shaft is only evidence for recognition; once
	// accepted, the geometry is deliberately exact and can never carry the
	// original bow through the snap.
	const shaftPts = synthSegment(tail, tip);
	const headT = shaftPts.length * 8;

	// Exact straight shaft, then left wing, back to tip, then right wing.
	return [
		...shaftPts,
		...synthSegment(tip, leftWing, headT),
		...synthSegment(leftWing, tip, headT + 100),
		...synthSegment(tip, rightWing, headT + 200),
	];
}

/**
 * Arrow recognition for open strokes
 *
 *  1. The tail is simply where the stroke started (body[0]). The tip is
 *     the point FARTHEST from the tail. A real arrowhead's wings sweep
 *     BACKWARD, toward the tail, so they sit closer to the tail than the
 *     tip does - the tip search is naturally immune to a big or lopsided
 *     arrowhead hijacking it.
 *  2. No straightness test on the shaft. A hand-drawn shaft that bows or
 *     kinks is still obviously meant as a straight arrow - the synthesized
 *     result is always a clean line. Instead, a single sanity check
 *     replaces it: total path length versus tail-to-tip distance. A shaft
 *     (even a bowed one) plus a normal arrowhead retraces only a little;
 *     an actual scribble backtracks constantly and blows well past that.
 *  3. Ink spatially near the tip (within ARROW_TIP_RADIUS of shaft length)
 *     counts as a wing once it deviates PERPENDICULARLY from the tail-tip
 *     axis by at least ARROW_FLUTTER_MIN (fraction of shaft length). Both
 *     sides of the head have to clear the flutter bar; this keeps ordinary
 *     scribbles and end-of-line wrist jitter freehand.
 *
 * The shaft length gate matches MIN_PATH_LENGTH; a short flutter on a short
 * stroke is too ambiguous to classify as an arrow.
 */
const ARROW_TIP_RADIUS = 0.62; // fraction of shaft length = "near the tip"
const ARROW_FLUTTER_MIN = 0.012; // min perp deviation fraction = arrowhead wing
// A purely relative flutter bar is fine for a long shaft but nearly free on a
// short one: 3% of a 30-unit line is under 1px, so ordinary hand tremor at
// the very end of an intentional short line crossed it and every such line
// snapped to an arrow it was never meant to be. A wing has to clear this
// many world units regardless of how short the shaft is.
const ARROW_FLUTTER_MIN_ABS = 2.5;
// Wing evidence is the PEAK deviation on each side of the axis within the
// tip zone, not a run of consecutive same-sign samples: a real arrowhead
// drawn quickly flutters back and forth almost every sample, so consecutive
// samples legitimately alternate sign. A single noisy sample still can't
// fake a wing on its own: it would have to clear the flutter bar on BOTH
// sides of the axis, which ordinary jitter essentially never does, and the
// whole shaft still has to pass the path-ratio scribble check below.
const ARROW_MAX_PATH_RATIO = 3.8; // pathLength / shaftLen ceiling before it's "wandering", not an arrow
const ARROW_TIP_HYSTERESIS = 2; // world units a candidate must clear the current tip by to replace it
const ARROW_SHAFT_STRAIGHTNESS = 0.1; // max shaft-only perpendicular deviation, fraction of shaftLen

interface WingScan {
	tail: P;
	tip: P;
	tipIndex: number;
	shaftLen: number;
	shaftEndIdx: number;
	positiveWing: boolean;
	negativeWing: boolean;
}

/**
 * Shared tip + wing evidence gathering for arrow recognition. Returns null
 * when there isn't even enough shaft to consider (too short, or the ink
 * wandered too much to trust a tail-to-tip axis at all) - that's "no
 * evidence either way", not "definitely not an arrow", and callers treat
 * it as such.
 */
function scanForWings(body: readonly P[]): WingScan | null {
	if (body.length < 12) return null;

	// 1. Tail = where the stroke started. Tip = farthest point from it.
	//    A candidate has to beat the current farthest point by more than
	//    plain jitter before it takes over, so the true shaft end isn't
	//    discarded for a few tenths of a world unit of noise.
	const tail = body[0]!;
	let tip = tail;
	let tipIndex = 0;
	let shaftLen = 0;
	for (let i = 1; i < body.length; i++) {
		const d = dist(tail, body[i]!);
		if (d > shaftLen + ARROW_TIP_HYSTERESIS) {
			shaftLen = d;
			tip = body[i]!;
			tipIndex = i;
		}
	}
	if (shaftLen < MIN_PATH_LENGTH) return null;

	// 2. Reject genuine scribbles: a shaft (bowed or not) plus an
	//    arrowhead retraces only a little ground relative to how far it
	//    ultimately got from the tail.
	let traced = 0;
	for (let i = 1; i < body.length; i++) traced += dist(body[i - 1]!, body[i]!);
	if (traced > shaftLen * ARROW_MAX_PATH_RATIO) return null;

	// 3. Ink near the tip that strays off the tail-tip axis is a wing.
	//    Track the first index that enters the arrowhead zone so we can
	//    split the shaft from the head geometry for synthesis.
	const ux = (tip.x - tail.x) / shaftLen;
	const uy = (tip.y - tail.y) / shaftLen;
	const tipRadius = shaftLen * ARROW_TIP_RADIUS;
	const flutterThreshold = Math.max(shaftLen * ARROW_FLUTTER_MIN, ARROW_FLUTTER_MIN_ABS);
	let shaftEndIdx = body.length - 1;
	let maxPerp = -Infinity;
	let minPerp = Infinity;
	// Search the whole tip neighbourhood rather than relying on the exact
	// sample index at which the user touched the tip: on real pen input
	// the farthest sample is often one of the first wing samples, or the
	// user briefly overshoots the corner. Requiring BOTH sides of the axis
	// still prevents a bowed shaft / single-wing attempt from becoming an
	// arrow.
	for (let i = 1; i < body.length; i++) {
		const p = body[i]!;
		if (dist(p, tip) > tipRadius) continue;
		// First post-tip sample inside the arrowhead zone - shaft ends just
		// before the actual head. If the farthest sample itself is already a
		// wing, tipIndex remains the natural shaft endpoint.
		if (i > tipIndex && shaftEndIdx === body.length - 1) shaftEndIdx = Math.max(0, tipIndex);
		const vx = p.x - tail.x;
		const vy = p.y - tail.y;
		const signedPerp = vx * uy - vy * ux;
		if (signedPerp > maxPerp) maxPerp = signedPerp;
		if (signedPerp < minPerp) minPerp = signedPerp;
	}

	// 4. The shaft itself - everything OUTSIDE the tip zone - has to
	//    actually be a shaft: close to the tail-tip axis. A zigzag drawn
	//    at any scale reaches the tip zone's wing thresholds exactly the
	//    same way a real arrowhead does (see ARROW_SHAFT_STRAIGHTNESS's
	//    own comment), but its "shaft" is off-axis everywhere, not just
	//    at the very end. Skip if there is no shaft left to check - a
	//    tiny shaft is caught by MIN_PATH_LENGTH above already.
	let shaftDev = 0;
	let shaftSamples = 0;
	for (let i = 1; i < body.length; i++) {
		const p = body[i]!;
		if (dist(p, tip) <= tipRadius) continue;
		const vx = p.x - tail.x;
		const vy = p.y - tail.y;
		const perp = Math.abs(vx * uy - vy * ux);
		if (perp > shaftDev) shaftDev = perp;
		shaftSamples++;
	}
	if (shaftSamples > 0 && shaftDev > shaftLen * ARROW_SHAFT_STRAIGHTNESS) {
		return { tail, tip, tipIndex, shaftLen, shaftEndIdx, positiveWing: false, negativeWing: false };
	}

	return {
		tail,
		tip,
		tipIndex,
		shaftLen,
		shaftEndIdx,
		positiveWing: maxPerp > flutterThreshold,
		negativeWing: minPerp < -flutterThreshold,
	};
}

function classifyOpen(body: readonly P[]): SnapResult | null {
	// Try arrow first: an arrow is a strict superset of a line, so a line
	// that happens to have flutter at one end would otherwise win the wrong
	// shape. A real arrowhead has two sides - requiring one wing on each
	// side makes ordinary scribbles and end-of-line wrist jitter
	// overwhelmingly less likely to be rewritten as an arrow. Because the
	// scan starts after the tip, the check is also directional: a small
	// oscillation while approaching the tip cannot manufacture the missing
	// second wing.
	const scan = scanForWings(body);
	if (scan && scan.positiveWing && scan.negativeWing) {
		const shaftBody = body.slice(0, scan.shaftEndIdx + 1);
		return {
			kind: "arrow",
			points: synthArrow(shaftBody.length > 1 ? shaftBody : [scan.tail], scan.tip, scan.shaftLen),
		};
	}
	const a = body[0]!;
	const b = body[body.length - 1]!;
	const len = dist(a, b);
	if (len < MIN_PATH_LENGTH) return null;
	let worst = 0;
	for (const p of body) {
		const d = Math.abs((b.x - a.x) * (a.y - p.y) - (a.x - p.x) * (b.y - a.y)) / len;
		if (d > worst) worst = d;
	}
	if (worst > len * LINE_TOLERANCE) return null;

	// A single wing is ambiguous only when it is genuinely visible as a
	// separate hook. Tiny end jitter on an otherwise straight line must not
	// suppress the line snap. Use a tighter line-fit test for the one-wing
	// case: a clean line remains a line; a clearly hooked one-wing arrow
	// stays freehand.
	if (scan && (scan.positiveWing || scan.negativeWing)) {
		const ONE_WING_LINE_TOLERANCE = 0.045;
		if (worst > len * ONE_WING_LINE_TOLERANCE) return null;
	}
	return { kind: "line", points: synthSegment(a, b) };
}

function synthSegment(a: P, b: P, tOffset = 0): InkPoint[] {
	const len = dist(a, b);
	const n = Math.max(2, Math.ceil(len / SYNTH_STEP));
	const out: InkPoint[] = [];
	for (let i = 0; i <= n; i++) {
		const f = i / n;
		out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, pressure: 0.5, t: tOffset + i * 8 });
	}
	return out;
}

function synthPolygon(corners: readonly P[]): InkPoint[] {
	const out: InkPoint[] = [];
	let t = 0;
	for (let i = 0; i < corners.length; i++) {
		const a = corners[i]!;
		const b = corners[(i + 1) % corners.length]!;
		const len = dist(a, b);
		const n = Math.max(1, Math.ceil(len / SYNTH_STEP));
		for (let j = 0; j < n; j++) {
			const f = j / n;
			out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, pressure: 0.5, t });
			t += 8;
		}
	}
	// Close the outline and continue one step past the corner, so the end
	// cap buries in the first edge instead of meeting the start cap.
	out.push({ ...out[0]!, t });
	if (out.length > 2) out.push({ ...out[1]!, t: t + 8 });
	return out;
}

function synthEllipse(cx: number, cy: number, rx: number, ry: number): InkPoint[] {
	const circumference = Math.PI * (3 * (rx + ry) - Math.sqrt((3 * rx + ry) * (rx + 3 * ry)));
	const n = Math.max(16, Math.ceil(circumference / SYNTH_STEP));
	const out: InkPoint[] = [];
	// Two steps past full circle: start and end caps land mid-line instead
	// of meeting at the seam, which rendered as a notch at the top
	// (hardware, 2026-08-27).
	for (let i = 0; i <= n + 2; i++) {
		const a = (i / n) * Math.PI * 2 - Math.PI / 2;
		out.push({ x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry, pressure: 0.5, t: i * 8 });
	}
	return out;
}

/** Tuning lens: what the classifier saw. Diagnostics only. */
export function diagnoseShape(points: readonly InkPoint[]): Record<string, unknown> {
	const start = dwellStart(points);
	if (start === null) return { dwell: false };
	const body = points.slice(0, Math.max(start, 2)) as unknown as P[];
	const len = pathLength(body);
	const gap = dist(body[0]!, body[body.length - 1]!);
	const closed = gap < len * CLOSURE_FRACTION;
	const out: Record<string, unknown> = {
		dwell: true,
		bodyPts: body.length,
		len: Math.round(len),
		gapFrac: +(gap / len).toFixed(2),
		closed,
	};
	if (closed && body.length >= 8) {
		let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
		for (const q of body) {
			if (q.x < minX) minX = q.x;
			if (q.y < minY) minY = q.y;
			if (q.x > maxX) maxX = q.x;
			if (q.y > maxY) maxY = q.y;
		}
		const w = maxX - minX, h = maxY - minY;
		const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
		out.roundErr = +meanRadialError(body, cx, cy, w / 2, h / 2).toFixed(3);
		const N = 64, K = 3;
		const ring = resampleClosed(body, N);
		const turns: number[] = [];
		for (let i = 0; i < N; i++) {
			const a = ring[(i - K + N) % N]!;
			const c = ring[i]!;
			const b2 = ring[(i + K) % N]!;
			const a1 = Math.atan2(c.y - a.y, c.x - a.x);
			const a2 = Math.atan2(b2.y - c.y, b2.x - c.x);
			let t = Math.abs(a2 - a1);
			if (t > Math.PI) t = 2 * Math.PI - t;
			turns.push((t * 180) / Math.PI);
		}
		const sorted = [...turns].sort((x, y) => x - y);
		out.turnMedianDeg = +sorted[32]!.toFixed(0);
		out.turnPeaksDeg = turns
			.map((t, i) => ({ t, i }))
			.filter(({ t, i }) => {
				if (t <= 55) return false;
				for (let d = -4; d <= 4; d++) {
					if (d === 0) continue;
					const j = (i + d + N) % N;
					if (turns[j]! > t || (turns[j]! === t && j < i)) return false;
				}
				return true;
			})
			.map(({ t }) => Math.round(t));
	}
	return out;
}

/**
 * The whole gesture: a stroke that ended in a dwell either becomes its
 * clean shape or stays itself. Null = no dwell, or nothing recognizable.
 */
export function snapStroke(stroke: InkStroke, dwellConfirmed = false): InkStroke | null {
	// The stroke builder's min-distance filter DISCARDS stationary samples,
	// so a real hold leaves no tail in the stored points - dwell detection
	// from points alone missed nearly every genuine hold (diagnosed on 95
	// hardware strokes, 2026-08-27). The overlay now measures the hold at
	// the raw input layer and passes dwellConfirmed; the points-based path
	// below survives for synthetic tests and any caller without raw timing.
	let body: InkPoint[];
	if (dwellConfirmed) {
		body = [...stroke.points];
	} else {
		const start = dwellStart(stroke.points);
		if (start === null) return null;
		body = stroke.points.slice(0, Math.max(start, 2));
	}
	if (body.length < 8) return null;
	const len = pathLength(body);
	if (len < MIN_PATH_LENGTH) return null;
	const gap = dist(body[0]!, body[body.length - 1]!);
	const closed = gap < len * CLOSURE_FRACTION;
	const result = closed ? classifyClosed(body, gap / (len + gap)) : classifyOpen(body);
	if (!result) return null;
	const width = stroke.width;
	return {
		id: newStrokeId(),
		tool: stroke.tool,
		color: stroke.color,
		width,
		points: result.points,
		bbox: computeBBox(result.points, width * 2),
		createdAt: stroke.createdAt,
		...(stroke.device === "mouse" ? { device: stroke.device } : {}),
		...(stroke.widthMode === "uniform" ? { widthMode: stroke.widthMode } : {}),
		...(stroke.pressureProfile === "exp7" ? { pressureProfile: stroke.pressureProfile } : {}),
	};
}
