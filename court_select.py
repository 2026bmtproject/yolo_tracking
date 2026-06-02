#!/usr/bin/env python3
"""
Interactive badminton court detection selector and fine-tuner.

Extracts the middle frame from a video, runs three detection methods (r, r2, r3),
shows the results side-by-side for the user to choose, then lets the user drag
the four outer court corners to fine-tune. Outputs a CSV of 16 key points.

Usage:
    python court_select.py <video> [output.csv] [--save-candidates] [--save-result]
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

import r  as det_r
import r2 as det_r2
import r3 as det_r3

# ── Court geometry (metres, identical to r/r2/r3) ────────────────────────────
H_LINES = np.array([0.00, 0.76, 4.72, 6.705, 8.685, 12.65, 13.41])
V_LINES = np.array([0.00, 0.46, 3.05, 5.64, 6.10])

OUTPUT_IDX = [
    (0, 0), (0, 4),   # 0,1 : top-left, top-right outer corners
    (6, 0), (6, 4),   # 2,3 : bottom-left, bottom-right outer corners
    (0, 1), (0, 3),   # 4,5 : top baseline × singles lines
    (6, 1), (6, 3),   # 6,7 : bottom baseline × singles lines
    (2, 0), (2, 4),   # 8,9 : top service line × sidelines
    (4, 0), (4, 4),   # 10,11: bottom service line × sidelines
    (2, 2), (4, 2),   # 12,13: center service line T-junctions
    (3, 0), (3, 4),   # 14,15: net × sidelines
]

# Indices within the 16-point list that are the four draggable outer corners
CORNER_IDXS = [0, 1, 2, 3]  # TL, TR, BL, BR

# Court-space (V, H) coordinates of the four corners in metres
CORNER_COURT_PTS = np.array([
    [V_LINES[0],  H_LINES[0]],   # TL
    [V_LINES[-1], H_LINES[0]],   # TR
    [V_LINES[0],  H_LINES[-1]],  # BL
    [V_LINES[-1], H_LINES[-1]],  # BR
], dtype=np.float32)

# Lines to draw (pairs of indices into the 16-point output list)
COURT_DRAW_LINES = [
    (0,  1),   # top baseline
    (2,  3),   # bottom baseline
    (0,  2),   # left sideline
    (1,  3),   # right sideline
    (4,  6),   # left singles line
    (5,  7),   # right singles line
    (8,  9),   # top service line
    (10, 11),  # bottom service line
    (12, 13),  # center service line
    (14, 15),  # net
]

DISPLAY_W   = 1280
DISPLAY_H   = 720
METHOD_NAMES = ["R1 (Fixed-Green ROI)", "R2 (Adaptive Color)", "R3 (Multi-ROI)"]
DETECTORS    = [det_r, det_r2, det_r3]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_middle_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read middle frame")
    return frame


def parse_court_point_line(line: str) -> list[float]:
    parts = [part.strip() for part in line.replace(",", ";").split(";") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid court point line: {line!r}")
    return [float(parts[0]), float(parts[1])]


def load_court_points_from_csv(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8-sig") as file:
        points = [parse_court_point_line(line) for line in file if line.strip()]

    if len(points) < 4:
        raise ValueError(f"{csv_path} must contain at least 4 court points")

    return np.array(points, dtype=np.int32)


def load_outer_corners_from_csv(csv_path: Path) -> np.ndarray:
    return load_court_points_from_csv(csv_path)[:4]


def scale_to_fit(img: np.ndarray, max_w: int, max_h: int):
    """Downscale img to fit within max_w×max_h; return (scaled_img, scale_factor)."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def recompute_from_corners(corners: np.ndarray):
    """
    Given 4 pixel positions [TL, TR, BL, BR] (shape 4×2 float32),
    compute the homography mapping court-space → pixel-space and project
    all 16 OUTPUT_IDX points.  Returns list of (x, y) tuples or None.
    """
    src_points = np.asarray(CORNER_COURT_PTS, dtype=np.float32)
    dst_points = np.asarray(corners, dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    if H is None:
        return None
    pts = []
    for hi, vi in OUTPUT_IDX:
        p = H @ np.array([V_LINES[vi], H_LINES[hi], 1.0])
        pts.append((float(p[0] / p[2]), float(p[1] / p[2])))
    return pts


def draw_court(img: np.ndarray, pts,
               line_color=(0, 220, 220),
               pt_color=(50, 255, 50),
               corner_color=(0, 140, 255),
               pt_r=4, corner_r=10,
               highlight_corners=True) -> np.ndarray:
    """Draw court boundary lines and key-point markers onto a copy of img."""
    out = img.copy()
    if pts is None:
        h, w = out.shape[:2]
        cv2.putText(out, "DETECTION FAILED",
                    (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out

    for i, j in COURT_DRAW_LINES:
        if i < len(pts) and j < len(pts):
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            p2 = (int(round(pts[j][0])), int(round(pts[j][1])))
            cv2.line(out, p1, p2, line_color, 1, cv2.LINE_AA)

    for idx, (x, y) in enumerate(pts):
        cx, cy = int(round(x)), int(round(y))
        if highlight_corners and idx in CORNER_IDXS:
            cv2.circle(out, (cx, cy), corner_r,     corner_color,    -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), corner_r + 2, (255, 255, 255),  1, cv2.LINE_AA)
        else:
            cv2.circle(out, (cx, cy), pt_r, pt_color, -1, cv2.LINE_AA)

    return out


# ── Phase 1: three-way comparison ────────────────────────────────────────────

def show_comparison(frame: np.ndarray, results: list,
                    save_candidates: bool, stem: str) -> int:
    """
    Show three detection results side-by-side.
    User clicks a panel or presses 1/2/3 to choose. Returns 0/1/2.
    """
    # Each panel preserves the frame's aspect ratio at 620px wide
    fh, fw    = frame.shape[:2]
    PANEL_W   = 620
    PANEL_H   = int(PANEL_W * fh / fw)
    BAR_H     = 50
    COMP_W    = PANEL_W * 3
    COMP_H    = PANEL_H + BAR_H

    panels = []
    for i, (name, pts) in enumerate(results):
        vis = draw_court(frame, pts, highlight_corners=False)

        lbl_color = (80, 255, 80) if pts is not None else (60, 60, 255)
        label = f"{i + 1}. {name}"
        cv2.putText(vis, label, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0),      3)
        cv2.putText(vis, label, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    lbl_color,      2)

        if save_candidates:
            path = f"{stem}_candidate_{i + 1}.jpg"
            cv2.imwrite(path, vis)
            print(f"  [saved] {path}")

        scaled, _ = scale_to_fit(vis, PANEL_W, PANEL_H)
        sh, sw = scaled.shape[:2]
        panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        panel[:sh, :sw] = scaled
        panels.append(panel)

    combined = np.zeros((COMP_H, COMP_W, 3), np.uint8)
    for i, p in enumerate(panels):
        combined[:PANEL_H, i * PANEL_W:(i + 1) * PANEL_W] = p
        if i > 0:
            combined[:PANEL_H, i * PANEL_W:i * PANEL_W + 1] = 80

    # Instruction bar below all panels
    combined[PANEL_H:, :] = (30, 30, 30)
    cv2.putText(combined,
                "Click a panel  or  press 1 / 2 / 3  to select the best result",
                (10, PANEL_H + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    selected = [-1]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and y < PANEL_H:
            idx = min(x // PANEL_W, 2)
            if results[idx][1] is not None:
                selected[0] = idx

    cv2.namedWindow("Select Detection Method", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Detection Method", COMP_W, COMP_H)
    cv2.setWindowProperty(
        "Select Detection Method",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    cv2.setMouseCallback("Select Detection Method", on_mouse)

    while selected[0] < 0:
        cv2.imshow("Select Detection Method", combined)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('1'), ord('2'), ord('3')):
            idx = key - ord('1')
            if results[idx][1] is not None:
                selected[0] = idx
        elif key == 27:          # ESC → first valid
            for i, (_, pts) in enumerate(results):
                if pts is not None:
                    selected[0] = i
                    break

    cv2.destroyWindow("Select Detection Method")
    return selected[0]


# ── Phase 2: interactive corner fine-tuning ───────────────────────────────────

def fine_tune(frame: np.ndarray, initial_pts: list) -> list:
    """
    Show the frame with draggable orange corner handles.
    Moving a corner recomputes the homography and updates all 16 key points.
    Press Enter to confirm or ESC to revert. Returns points in original-frame coords.
    """
    disp_frame, scale = scale_to_fit(frame, DISPLAY_W, DISPLAY_H)
    dh, dw = disp_frame.shape[:2]

    def to_disp(pt):
        return np.array([pt[0] * scale, pt[1] * scale], dtype=np.float32)

    def from_disp(pt):
        return (pt[0] / scale, pt[1] / scale)

    corners_disp = np.array(
        [to_disp(initial_pts[i]) for i in CORNER_IDXS], dtype=np.float32)

    state = {
        "corners": corners_disp.copy(),
        "pts":     recompute_from_corners(corners_disp),
        "drag":    -1,
        "done":    False,
        "cancel":  False,
    }

    HANDLE_R = 6   # display pixels — smaller handle circles
    HIT_R    = HANDLE_R * 3  # click-detection radius stays generous
    TEXT_H   = 36  # height of instruction bar below image

    def render():
        vis = draw_court(disp_frame, state["pts"],
                         corner_r=HANDLE_R,
                         pt_r=3,
                         highlight_corners=True)
        # Place image on top; instruction bar on a separate strip below
        canvas = np.zeros((dh + TEXT_H, dw, 3), np.uint8)
        canvas[:dh] = vis
        canvas[dh:] = (30, 30, 30)
        cv2.putText(canvas,
                    "Drag orange corners  |  Enter: confirm  |  ESC: cancel",
                    (8, dh + TEXT_H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        return canvas

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["pts"] is None or y >= dh:
                return
            best_k, best_d = -1, float("inf")
            for k, ci in enumerate(CORNER_IDXS):
                px = int(round(state["pts"][ci][0]))
                py = int(round(state["pts"][ci][1]))
                d  = np.hypot(x - px, y - py)
                if d < HIT_R and d < best_d:
                    best_d, best_k = d, k
            state["drag"] = best_k

        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] >= 0:
            state["corners"][state["drag"]] = [float(x), float(y)]
            state["pts"] = recompute_from_corners(state["corners"])

        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = -1

    cv2.namedWindow("Fine-tune Court Corners", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fine-tune Court Corners", dw, dh + TEXT_H)
    cv2.setWindowProperty(
        "Fine-tune Court Corners",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    cv2.setMouseCallback("Fine-tune Court Corners", on_mouse)

    while not state["done"] and not state["cancel"]:
        cv2.imshow("Fine-tune Court Corners", render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):   # Enter / newline
            state["done"] = True
        elif key == 27:        # ESC → revert
            state["cancel"] = True

    cv2.destroyWindow("Fine-tune Court Corners")

    if state["cancel"] or state["pts"] is None:
        return initial_pts

    return [from_disp(p) for p in state["pts"]]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Interactive badminton court detection selector + fine-tuner")
    ap.add_argument("video",
                    help="Input video file")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output CSV path (default: <video_stem>_court.csv)")
    ap.add_argument("--save-candidates", action="store_true",
                    help="Save the three candidate detection images as JPEGs")
    ap.add_argument("--save-result", action="store_true",
                    help="Save the final adjusted result image as a JPEG")
    args = ap.parse_args()

    video_path = Path(args.video)
    stem       = video_path.stem
    csv_path   = args.output or f"{stem}_court.csv"

    print(f"[1/4] Extracting middle frame from '{video_path.name}' ...")
    frame = extract_middle_frame(str(video_path))
    print(f"      Frame size: {frame.shape[1]}×{frame.shape[0]}")

    print("[2/4] Running three detection methods ...")
    results = []
    for name, det in zip(METHOD_NAMES, DETECTORS):
        try:
            pts, _ = det.detect_from_frame(frame)
            print(f"      {name}: OK  ({len(pts)} points)")
            results.append((name, pts))
        except Exception as e:
            print(f"      {name}: FAILED — {e}")
            results.append((name, None))

    if all(pts is None for _, pts in results):
        sys.exit("Error: all three detectors failed — cannot continue.")

    print("[3/4] Showing comparison — click or press 1/2/3 to select ...")
    sel       = show_comparison(frame, results, args.save_candidates, stem)
    sel_name, sel_pts = results[sel]
    print(f"      Selected: {sel_name}")

    print("[4/4] Fine-tuning — drag orange corners, then press Enter ...")
    final_pts = fine_tune(frame, sel_pts)

    with open(csv_path, "w") as f:
        for x, y in final_pts:
            f.write(f"{x:.4f};{y:.4f}\n")
    print(f"\n[done] {len(final_pts)} points  →  {csv_path}")

    if args.save_result:
        vis         = draw_court(frame, final_pts, highlight_corners=False)
        result_path = f"{stem}_result.jpg"
        cv2.imwrite(result_path, vis)
        print(f"[done] result image  →  {result_path}")


if __name__ == "__main__":
    main()
