#!/usr/bin/env python3
"""
Badminton court boundary detection — multi-ROI version.

Compared with r2.py:
  • Court colour detection returns multiple hue peaks instead of a single peak.
  • Court ROI generation is done per candidate colour, so multi-colour
    courts can be handled without assuming one dominant hue.
  • Candidate ROIs are filtered by quick line-count checks before the
    expensive full homography fit.
  • ROI scoring uses white-line density as a supporting cue.

Core logic stays intact:
  • line refinement
  • line support filtering
  • homography fitting / scoring
  • court geometry / OUTPUT_IDX
"""

import sys
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

# ── Court geometry (metres) ───────────────────────────────────────────────────
H_LINES = np.array([0.00, 0.76, 4.72, 6.705, 8.685, 12.65, 13.41])
V_LINES = np.array([0.00, 0.46, 3.05, 5.64, 6.10])

OUTPUT_IDX = [
    (0, 0), (0, 4),   # top baseline corners
    (6, 0), (6, 4),   # bottom baseline corners
    (0, 1), (0, 3),   # top baseline × singles lines
    (6, 1), (6, 3),   # bottom baseline × singles lines
    (2, 0), (2, 4),   # top service line × sidelines
    (4, 0), (4, 4),   # bottom service line × sidelines
    (2, 2), (4, 2),   # center service line
    (3, 0), (3, 4),   # net × sidelines
]


# ── Stage 0: Court ROI from candidate court-colour connected components ──────

def _detect_court_colors(frame: np.ndarray, hsv: np.ndarray = None, top_k: int = 3):
    """
    Detect multiple dominant court-colour hue peaks from a center-bottom
    sample box of the frame.

    Returns a list of candidate HSV thresholds:
        [(peak_hue, h_tol, s_lo, v_lo), ...]
    sorted from strongest to weakest peak.
    """
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_img, w_img = frame.shape[:2]
    # Sample box: middle 50 % wide, between 45-75 % of height.
    # This avoids the audience above and the yellow apron at the bottom.
    cx0, cx1 = w_img // 4, w_img * 3 // 4
    cy0, cy1 = int(h_img * 0.45), int(h_img * 0.75)
    sample = hsv[cy0:cy1, cx0:cx1]
    if sample.size == 0:
        return []

    sat_mask = (sample[..., 1] > 50) & (sample[..., 2] > 40)
    if sat_mask.sum() < sample.shape[0] * sample.shape[1] * 0.10:
        return []

    hue = sample[sat_mask, 0].astype(np.int32)
    hist = np.bincount(hue, minlength=180).astype(np.float32)
    pad = 10
    padded = np.r_[hist[-pad:], hist, hist[:pad]]
    kernel = np.ones(2 * pad + 1, np.float32) / (2 * pad + 1)
    smooth = np.convolve(padded, kernel, mode='same')[pad:-pad]

    max_peak = float(smooth.max())
    if max_peak <= 0:
        return []

    # Greedy peak picking with circular suppression so we can return
    # top-2 / top-3 hue candidates even when the sample mixes court colours.
    min_sep = 12
    selected = []
    order = np.argsort(smooth)[::-1]
    for idx in order:
        score = float(smooth[idx])
        if score < max_peak * 0.18:
            break
        if any(min((idx - p) % 180, (p - idx) % 180) < min_sep for p in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= top_k:
            break

    # ± h_tol around the peak.  30° covers most broadcast lighting variation
    # without admitting nearby unrelated colours.
    h_tol = 30
    s_lo = 40
    v_lo = 35
    return [(peak_hue, h_tol, s_lo, v_lo) for peak_hue in selected]


def _detect_court_color(frame: np.ndarray, hsv: np.ndarray = None):
    """Backward-compatible single-candidate wrapper."""
    colors = _detect_court_colors(frame, hsv=hsv, top_k=1)
    return colors[0] if colors else None


def _inrange_hue(hsv: np.ndarray,
                 peak_hue: int, h_tol: int,
                 s_lo: int, v_lo: int) -> np.ndarray:
    """HSV threshold around `peak_hue` ± `h_tol`, with red-wrap support."""
    lo = peak_hue - h_tol
    hi = peak_hue + h_tol
    if lo < 0:
        m1 = cv2.inRange(hsv, np.array([(180 + lo) % 180, s_lo, v_lo]),
                         np.array([180, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]),
                         np.array([hi, 255, 255]))
        return cv2.bitwise_or(m1, m2)
    if hi >= 180:
        m1 = cv2.inRange(hsv, np.array([lo, s_lo, v_lo]),
                         np.array([179, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]),
                         np.array([hi - 180, 255, 255]))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, np.array([lo, s_lo, v_lo]),
                       np.array([hi, 255, 255]))


def _build_roi_from_mask(mask: np.ndarray,
                         min_area_ratio: float,
                         dilate_px: int):
    """Convert a colour mask into roi_loose / roi_tight / roi_core."""
    h_img, w_img = mask.shape[:2]

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, None, None, None

    best_label, best_score = -1, -1.0
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < h_img * w_img * min_area_ratio:
            continue
        cy = cents[lbl, 1]
        score = area * max(0.05, cy / h_img)
        if score > best_score:
            best_score, best_label = score, lbl
    if best_label < 0:
        return None, None, None, None

    court = (labels == best_label).astype(np.uint8) * 255

    # Re-merge directly-adjacent same-coloured components.
    main_y0 = stats[best_label, cv2.CC_STAT_TOP]
    main_y1 = main_y0 + stats[best_label, cv2.CC_STAT_HEIGHT]
    main_x0 = stats[best_label, cv2.CC_STAT_LEFT]
    main_x1 = main_x0 + stats[best_label, cv2.CC_STAT_WIDTH]
    main_area = stats[best_label, cv2.CC_STAT_AREA]
    for lbl in range(1, n):
        if lbl == best_label:
            continue
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < main_area * 0.03:
            continue
        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        y0 = stats[lbl, cv2.CC_STAT_TOP]
        x1 = x0 + stats[lbl, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[lbl, cv2.CC_STAT_HEIGHT]
        h_overlap = min(x1, main_x1) - max(x0, main_x0)
        v_gap = max(0, max(main_y0, y0) - min(main_y1, y1))
        if h_overlap > 0.3 * min(main_x1 - main_x0, x1 - x0) and v_gap == 0:
            court = cv2.bitwise_or(court, (labels == lbl).astype(np.uint8) * 255)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    court = cv2.morphologyEx(court, cv2.MORPH_CLOSE, k_close, iterations=1)

    contours, _ = cv2.findContours(court, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    biggest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(biggest)
    roi_core = np.zeros_like(court)
    cv2.fillConvexPoly(roi_core, hull, 255)

    k_loose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (dilate_px, dilate_px))
    roi_loose = cv2.dilate(roi_core, k_loose, iterations=1)
    tight_px = max(3, dilate_px // 4)
    k_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (tight_px, tight_px))
    roi_tight = cv2.dilate(roi_core, k_tight, iterations=1)

    return roi_loose, roi_tight, roi_core, best_score


def detect_court_roi(frame: np.ndarray,
                     min_area_ratio: float = 0.04,
                     dilate_px: int = 10,
                     gray: np.ndarray = None,
                     line_density: np.ndarray = None,
                     top_k: int = 3):
    """
    Find candidate playing-surface ROIs from multiple court-colour peaks.

    Returns a list of (roi_loose, roi_tight, roi_core) tuples ordered by
    a colour + line-density support score.
    """
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if line_density is None:
        line_pixels = detect_court_pixels(gray)
        line_density = cv2.blur(line_pixels, (31, 31))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colors = _detect_court_colors(frame, hsv=hsv, top_k=top_k)
    if not colors:
        return []

    scored_candidates = []
    for peak_hue, h_tol, s_lo, v_lo in colors:
        mask = _inrange_hue(hsv, peak_hue, h_tol, s_lo, v_lo)
        roi_loose, roi_tight, roi_core, base_score = _build_roi_from_mask(
            mask, min_area_ratio=min_area_ratio, dilate_px=dilate_px)
        if roi_loose is None:
            continue

        # Step 3: quick line-count filter per ROI.
        masked_gray = cv2.bitwise_and(gray, roi_loose)
        binary = detect_court_pixels(masked_gray)
        lines = detect_lines(binary)
        if len(lines) < 5:
            continue

        # Step 6: white-line density as an auxiliary score.
        roi_mask = roi_loose > 0
        density_bonus = float(line_density[roi_mask].mean()) if np.any(roi_mask) else 0.0
        scored_candidates.append(
            (base_score + density_bonus, (roi_loose, roi_tight, roi_core)))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [roi_tuple for _, roi_tuple in scored_candidates]


# ── Stage 1: Court-line pixel detection ───────────────────────────────────────

def detect_court_pixels(gray: np.ndarray,
                        lum_thresh: int = 80,
                        diff_thresh: int = 20,
                        t: int = 10) -> np.ndarray:
    g = gray.astype(np.float32)
    h, w = g.shape

    gph = np.pad(g, ((0, 0), (t, t)), mode='edge')
    horiz = ((g - gph[:, :w] > diff_thresh)
             & (g - gph[:, 2 * t: 2 * t + w] > diff_thresh))

    gpv = np.pad(g, ((t, t), (0, 0)), mode='edge')
    vert = ((g - gpv[:h, :] > diff_thresh)
            & (g - gpv[2 * t: 2 * t + h, :] > diff_thresh))

    out = np.zeros_like(gray)
    out[(gray > lum_thresh) & (horiz | vert)] = 255
    return out


def filter_structure_tensor(gray: np.ndarray,
                            binary: np.ndarray,
                            ksize: int = 21,
                            ratio_thresh: float = 4.0) -> np.ndarray:
    k = ksize | 1
    sigma = k / 6.0
    g = gray.astype(np.float32)

    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.GaussianBlur(Ix * Ix, (k, k), sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (k, k), sigma)
    Iyy = cv2.GaussianBlur(Iy * Iy, (k, k), sigma)

    disc = np.sqrt(np.maximum(((Ixx - Iyy) / 2) ** 2 + Ixy ** 2, 0.0))
    lam1 = (Ixx + Iyy) / 2 + disc
    lam2 = (Ixx + Iyy) / 2 - disc
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(lam2) > 1e-6, lam1 / lam2, 0.0)

    out = np.zeros_like(binary)
    out[(binary > 0) & (ratio > ratio_thresh)] = 255
    return out


# ── Stage 1b: line filtering by ROI ──────────────────────────────────────────

def filter_lines_by_support(lines: list, white_pts: np.ndarray,
                            roi_core: np.ndarray,
                            min_support: int = 80,
                            band_px: float = 4.0,
                            min_core_ratio: float = 0.5,
                            max_thickness: float = 1.6) -> list:
    """
    Keep only lines that satisfy ALL THREE:
      • At least `min_support` white pixels lie within `band_px`
        perpendicular distance of the line.
      • At least `min_core_ratio` of those supporting pixels lie inside
        `roi_core` (the un-dilated convex hull of the court colour).
      • The supporting pixels form a *thin* band: their perpendicular-
        distance standard deviation is below `max_thickness` px.
    """
    if white_pts.size == 0 or roi_core is None:
        return lines
    h_img, w_img = roi_core.shape[:2]
    kept = []
    for line in lines:
        dp = white_pts - line.pt
        perp = dp[:, 0] * line.dv[1] - dp[:, 1] * line.dv[0]
        near_mask = np.abs(perp) < band_px
        if near_mask.sum() < min_support:
            continue
        near = white_pts[near_mask]
        ix = np.clip(near[:, 0].astype(int), 0, w_img - 1)
        iy = np.clip(near[:, 1].astype(int), 0, h_img - 1)
        if (roi_core[iy, ix] > 0).mean() < min_core_ratio:
            continue
        thickness = float(np.std(perp[near_mask]))
        if thickness > max_thickness:
            continue
        kept.append(line)
    return kept


# ── Stage 2: Line detection ───────────────────────────────────────────────────

class Line:
    def __init__(self, pt, dv):
        self.pt = np.asarray(pt, float)
        n = np.linalg.norm(dv)
        self.dv = np.asarray(dv, float) / n if n > 1e-9 else np.asarray(dv, float)

    def perpendicular_distance(self, p):
        dp = np.asarray(p) - self.pt
        return abs(dp[0] * self.dv[1] - dp[1] * self.dv[0])

    def angle_with(self, other):
        return np.arccos(np.clip(abs(np.dot(self.dv, other.dv)), 0.0, 1.0))

    def intersect(self, other):
        da, db = self.dv, other.dv
        dp = other.pt - self.pt
        denom = da[0] * db[1] - da[1] * db[0]
        if abs(denom) < 1e-9:
            return None
        t = (dp[0] * db[1] - dp[1] * db[0]) / denom
        return self.pt + t * da

    @classmethod
    def from_segment(cls, x1, y1, x2, y2):
        return cls([x1, y1], [x2 - x1, y2 - y1])


def detect_lines(binary: np.ndarray, hough_thresh: int = 50) -> list:
    segs = cv2.HoughLinesP(binary, 1, np.pi / 180, hough_thresh,
                           minLineLength=50, maxLineGap=10)
    if segs is None:
        return []
    return [Line.from_segment(*s[0]) for s in segs
            if not (s[0][0] == s[0][2] and s[0][1] == s[0][3])]


def refine_lines(lines: list, white_pts: np.ndarray,
                 dist_thresh: float = 8.0, iterations: int = 50) -> list:
    if white_pts.size == 0:
        return lines
    refined = []
    for line in lines:
        pt, dv = line.pt.copy(), line.dv.copy()
        for _ in range(iterations):
            dp = white_pts - pt
            dists = np.abs(dp[:, 0] * dv[1] - dp[:, 1] * dv[0])
            near = white_pts[dists < dist_thresh]
            if len(near) < 2:
                break
            vx, vy, x0, y0 = cv2.fitLine(
                near.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            new_pt = np.array([x0, y0])
            new_dv = np.array([vx, vy])
            n = np.linalg.norm(new_dv)
            if n > 1e-9:
                new_dv /= n
            if np.allclose(pt, new_pt, atol=0.5) and np.allclose(dv, new_dv, atol=1e-3):
                break
            pt, dv = new_pt, new_dv
        refined.append(Line(pt, dv))
    return refined


def deduplicate_lines(lines: list,
                      angle_thresh: float = np.deg2rad(8),
                      dist_thresh: float = 10.0) -> list:
    """
    Merge near-duplicate lines.  Smaller distance threshold than the
    original (10 px vs 20 px) — at the top of the court the back baseline,
    long service line and net top can sit within 15 px of each other in
    image space due to perspective compression, and merging them all
    leaves us short of horizontal lines.
    """
    used = [False] * len(lines)
    out = []
    for i, la in enumerate(lines):
        if used[i]:
            continue
        group = [la]
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            lb = lines[j]
            if (la.angle_with(lb) < angle_thresh
                    and lb.perpendicular_distance(la.pt) < dist_thresh):
                group.append(lb)
                used[j] = True
        avg_pt = np.mean([l.pt for l in group], axis=0)
        avg_dv = np.mean([l.dv for l in group], axis=0)
        out.append(Line(avg_pt, avg_dv))
    return out


def classify_and_sort(lines: list):
    h_lines, v_lines = [], []
    for l in lines:
        a = abs(np.arctan2(l.dv[1], l.dv[0]))
        (h_lines if (a < np.pi / 4 or a > 3 * np.pi / 4) else v_lines).append(l)
    h_lines.sort(key=lambda l: l.pt[1])
    v_lines.sort(key=lambda l: l.pt[0])
    return h_lines, v_lines


# ── Stage 3: Court model fitting ──────────────────────────────────────────────

def _homography_from_corners(h0, h1, v0, v1):
    corners = [h0.intersect(v0), h0.intersect(v1),
               h1.intersect(v0), h1.intersect(v1)]
    if any(c is None for c in corners):
        return None
    src = np.float32(corners)
    dst = np.float32([[V_LINES[0],  H_LINES[0]],
                      [V_LINES[-1], H_LINES[0]],
                      [V_LINES[0],  H_LINES[-1]],
                      [V_LINES[-1], H_LINES[-1]]])
    H, _ = cv2.findHomography(dst, src)
    return H


def _project_outer_corners(H: np.ndarray):
    """
    Return the 4 outer court corners in **polygon order** (clockwise):
        TL, TR, BR, BL
    so that cv2.contourArea / cv2.fillConvexPoly produce correct results.
    """
    corners = []
    for (cx, cy) in [(V_LINES[0],  H_LINES[0]),
                     (V_LINES[-1], H_LINES[0]),
                     (V_LINES[-1], H_LINES[-1]),
                     (V_LINES[0],  H_LINES[-1])]:
        p = H @ np.array([cx, cy, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        corners.append(p[:2] / p[2])
    return np.array(corners)


def _is_valid_homography(H: np.ndarray, img_shape: tuple,
                         roi_mask: np.ndarray = None) -> bool:
    """
    Stricter than the original:
    1. Top baseline above bottom baseline.
    2. Outer corners within ±40 % of image bounds.
    3. Projected court area between 5 % and 90 % of image.
    4. Aspect-ratio sanity for a TV-camera view.
    5. Trapezoid sanity (top vs bottom, left vs right).
    6. Centre of the projected court must lie inside the ROI.
    """
    h_img, w_img = img_shape[:2]
    corners = _project_outer_corners(H)
    if corners is None:
        return False
    TL, TR, BR, BL = corners

    if (TL[1] + TR[1]) >= (BL[1] + BR[1]):
        return False

    mx, my = w_img * 0.4, h_img * 0.4
    if (corners[:, 0].min() < -mx or corners[:, 0].max() > w_img + mx or
            corners[:, 1].min() < -my or corners[:, 1].max() > h_img + my):
        return False

    area = abs(cv2.contourArea(corners.astype(np.float32)))
    if area < w_img * h_img * 0.05 or area > w_img * h_img * 0.9:
        return False

    top_w   = np.linalg.norm(TR - TL)
    bot_w   = np.linalg.norm(BR - BL)
    left_h  = np.linalg.norm(BL - TL)
    right_h = np.linalg.norm(BR - TR)
    avg_w = (top_w + bot_w) / 2
    avg_h = (left_h + right_h) / 2
    if avg_w < 20 or avg_h < 20:
        return False
    aspect = avg_h / avg_w
    if aspect < 0.3 or aspect > 4.0:
        return False

    if max(top_w, bot_w) > 3.5 * min(top_w, bot_w):
        return False
    if max(left_h, right_h) > 2.5 * min(left_h, right_h):
        return False

    if roi_mask is not None:
        cx, cy = corners.mean(axis=0)
        cxi, cyi = int(round(cx)), int(round(cy))
        if not (0 <= cxi < w_img and 0 <= cyi < h_img):
            return False
        if roi_mask[cyi, cxi] == 0:
            return False

    return True


def _score_homography(H: np.ndarray, all_lines: list,
                      roi_mask: np.ndarray = None,
                      img_shape: tuple = None,
                      tol: float = 0.8) -> float:
    """
    Line-alignment count, multiplied by an IoU factor against the
    court-coloured ROI.
    """
    if H is None:
        return 0.0
    Hi = np.linalg.inv(H)

    def to_court(p):
        c = Hi @ np.array([p[0], p[1], 1.0])
        return c[:2] / (c[2] + 1e-9)

    align = 0
    for line in all_lines:
        pts = [to_court(line.pt + line.dv * s) for s in (-200, 0, 200)]
        ys = [p[1] for p in pts]; xs = [p[0] for p in pts]
        if max(ys) - min(ys) < tol and np.any(np.abs(H_LINES - np.mean(ys)) < tol):
            align += 1; continue
        if max(xs) - min(xs) < tol and np.any(np.abs(V_LINES - np.mean(xs)) < tol):
            align += 1

    iou = 1.0
    if roi_mask is not None and img_shape is not None:
        h_img, w_img = img_shape[:2]
        corners = _project_outer_corners(H)
        if corners is not None:
            quad = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillConvexPoly(quad, corners.astype(np.int32), 255)
            quad_area = np.count_nonzero(quad)
            roi_area = np.count_nonzero(roi_mask)
            inter = np.count_nonzero(cv2.bitwise_and(quad, roi_mask))
            union = quad_area + roi_area - inter
            if union > 0:
                iou = inter / union

    return align * (0.2 + 0.8 * iou)


def fit_court(h_lines: list, v_lines: list, img_shape: tuple,
              roi_mask: np.ndarray = None):
    """
    Try every pair of H- and V-lines as outer baselines/sidelines.
    """
    best, best_H = -1.0, None
    all_lines = h_lines + v_lines

    max_h_span = max((h_lines[-1].pt[1] - h_lines[0].pt[1]), 1.0)
    max_v_span = max((v_lines[-1].pt[0] - v_lines[0].pt[0]), 1.0)

    for (i, j) in combinations(range(len(h_lines)), 2):
        for (p, q) in combinations(range(len(v_lines)), 2):
            H = _homography_from_corners(h_lines[i], h_lines[j],
                                         v_lines[p], v_lines[q])
            if H is None or not _is_valid_homography(H, img_shape, roi_mask):
                continue
            s = _score_homography(H, all_lines, roi_mask, img_shape)
            h_span = (h_lines[j].pt[1] - h_lines[i].pt[1]) / max_h_span
            v_span = (v_lines[q].pt[0] - v_lines[p].pt[0]) / max_v_span
            weighted = s * (1.0 + 0.4 * (h_span + v_span))
            if weighted > best:
                best, best_H = weighted, H
    return best_H, best


# ── Stage 4: Project court points ─────────────────────────────────────────────

def project_court_points(H: np.ndarray) -> dict:
    pts = {}
    for hi, y in enumerate(H_LINES):
        for vi, x in enumerate(V_LINES):
            p = H @ np.array([x, y, 1.0])
            pts[(hi, vi)] = (float(p[0] / p[2]), float(p[1] / p[2]))
    return pts


def _run_pipeline_with_roi(frame: np.ndarray,
                           gray: np.ndarray,
                           binary: np.ndarray,
                           roi_tuple):
    """Run the original line → fit pipeline against a single ROI."""
    roi_loose, roi_tight, roi_core = roi_tuple

    filtered = filter_structure_tensor(gray, binary)
    if np.count_nonzero(filtered) < 500:
        filtered = binary

    if roi_loose is not None:
        filtered = cv2.bitwise_and(filtered, roi_loose)

    wy, wx = np.where(filtered > 0)
    white_pts = (np.column_stack([wx, wy]).astype(np.float32)
                 if wx.size else np.empty((0, 2), np.float32))

    lines = detect_lines(filtered)
    lines = refine_lines(lines, white_pts)
    lines = deduplicate_lines(lines)
    support_filtered = filter_lines_by_support(lines, white_pts, roi_core)
    lines = support_filtered if len(support_filtered) >= 4 else lines

    h_lines, v_lines = classify_and_sort(lines)
    if len(h_lines) < 2 or len(v_lines) < 2:
        raise RuntimeError(
            f"Not enough lines: {len(h_lines)}H, {len(v_lines)}V")

    H_mat, score = fit_court(h_lines, v_lines, frame.shape, roi_loose)
    if H_mat is None:
        raise RuntimeError("Court model fitting failed")

    all_pts = project_court_points(H_mat)
    out_pts = [all_pts[k] for k in OUTPUT_IDX if k in all_pts]
    return {
        "out_pts": out_pts,
        "score": score,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "all_pts": all_pts,
        "roi_mask": roi_loose,
        "roi_tight": roi_tight,
        "roi_core": roi_core,
        "H": H_mat,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_from_frame(frame: np.ndarray):
    """Run the full pipeline on a single frame; return (out_pts, debug_dict)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    line_pixels = detect_court_pixels(gray)
    line_density = cv2.blur(line_pixels, (31, 31))

    roi_candidates = detect_court_roi(
        frame, gray=gray, line_density=line_density)

    # Fall back to the full frame if no colour candidate survives.
    if not roi_candidates:
        roi_candidates = [(None, None, None)]

    best_result = None
    best_score = -np.inf
    for roi_tuple in roi_candidates:
        try:
            result = _run_pipeline_with_roi(frame, gray, line_pixels, roi_tuple)
        except RuntimeError:
            continue
        if result["score"] > best_score:
            best_result = result
            best_score = result["score"]

    if best_result is None and roi_candidates != [(None, None, None)]:
        best_result = _run_pipeline_with_roi(frame, gray, line_pixels, (None, None, None))

    if best_result is None:
        raise RuntimeError("Court model fitting failed")

    return best_result["out_pts"], best_result


def process(video_path: str, output_path: str = None, debug: bool = False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read middle frame")

    out_pts, dbg = detect_from_frame(frame)
    print(f"[detect] score={dbg['score']:.2f}  "
          f"H-lines={len(dbg['h_lines'])}  V-lines={len(dbg['v_lines'])}")

    if debug:
        vis = frame.copy()
        if dbg["roi_mask"] is not None:
            roi_overlay = np.zeros_like(vis)
            roi_overlay[:, :, 1] = dbg["roi_mask"] // 3
            vis = cv2.addWeighted(vis, 1.0, roi_overlay, 0.4, 0)
        for (x, y) in dbg["all_pts"].values():
            cv2.circle(vis, (int(x), int(y)), 3, (0, 200, 0), -1)
        for i, (x, y) in enumerate(out_pts):
            cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), 2)
            cv2.putText(vis, str(i + 1), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for l in dbg["h_lines"] + dbg["v_lines"]:
            p1 = tuple((l.pt - l.dv * 1500).astype(int))
            p2 = tuple((l.pt + l.dv * 1500).astype(int))
            cv2.line(vis, p1, p2, (255, 120, 0), 1)
        path = Path(video_path).with_suffix(".debug.jpg")
        cv2.imwrite(str(path), vis)
        print(f"[debug]  → {path}")

    if output_path:
        with open(output_path, "w") as f:
            for x, y in out_pts:
                f.write(f"{x:.4f};{y:.4f}\n")
        print(f"[output] {len(out_pts)} points → {output_path}")
    else:
        for i, (x, y) in enumerate(out_pts, 1):
            print(f"P{i:02d}: {x:.2f};{y:.2f}")

    return out_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    try:
        process(args.video, args.output, args.debug)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()