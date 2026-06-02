from pathlib import Path
import csv
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from court_select import load_outer_corners_from_csv


# =============================================================================
# 使用方式
# =============================================================================
# 1. 單支影片：
#    RUN_MODE = "single"
#    設定 VIDEO_SOURCE 和 OUTPUT_PATH。
#
# 2. 整個資料夾：
#    RUN_MODE = "folder"
#    設定 INPUT_DIR 和 OUTPUT_DIR。
#    輸出檔名會自動變成：原始檔名_player_only.mp4。
#
# 3. 執行：
#    .\yolo26_env\Scripts\python.exe badminton_pose_track.py
#
# 影片中出現橘色框代表 crop recovery 補偵測成功。
# 換攝影機角度或球場位置時，請重新標定球場或修改 COURT_POLYGON。

# =============================================================================
# 1. 操作區：通常只需要改這裡
# =============================================================================
MODEL_PATH = "yolo26m-pose.pt"

# "single"：處理 VIDEO_SOURCE 這一支影片。
# "folder"：處理 INPUT_DIR 裡所有影片。
RUN_MODE = "single"

# single 模式使用。
VIDEO_SOURCE = Path("test.mp4")
OUTPUT_PATH = Path("runs/player_only/test.mp4")

# folder 模式使用。
INPUT_DIR = Path("andersclip")
OUTPUT_DIR = Path("anders_fina_track")

# 輸出模式：
# "video"：只輸出處理後的影片。
# "csv"：只輸出骨架資料 CSV。
# "both"：兩者都輸出。
OUTPUT_MODE = "both"

# 是否先開啟 court_select.py 讓你手動標球場。
# True：執行時會開啟標點工具。
# False：直接使用下面 COURT_POLYGON 的四個點。
USE_COURT_SELECT = True

# True：每次都重新標球場。
# False：如果已經有球場標定檔，就直接沿用。
FORCE_COURT_SELECT = True


# =============================================================================
# 2. YOLO 與追蹤設定
# =============================================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
PROGRESS_BAR_WIDTH = 30
PROGRESS_UPDATE_FRAMES = 5

CONFIDENCE = 0.15
IMAGE_SIZE = 1280
DEVICE = 0
TRACKER = "badminton_tracker.yaml"


# =============================================================================
# 3. 球場標定設定
# =============================================================================
COURT_SELECT_SCRIPT = Path("court_select.py")
COURT_SELECT_OUTPUT_PATH = Path("runs/player_only/selected_court.csv")

# 手動球場四角座標，順序固定為：左上、右上、左下、右下。
COURT_POLYGON = np.array(
    [
        [270, 183],
        [592, 183],
        [132, 447],
        [728, 446],
    ],
    dtype=np.int32,
)
MANUAL_COURT_POLYGON = COURT_POLYGON.copy()


# =============================================================================
# 4. 選手篩選與補偵測設定
# =============================================================================
MAX_PLAYERS = 2
DRAW_COURT_ROI = True

# 球員腳點允許離球場多遠仍算候選人。
COURT_MARGIN_PIXELS = 50
COURT_TOP_MARGIN_PIXELS = 90
COURT_SIDE_MARGIN_PIXELS = 50
COURT_BOTTOM_MARGIN_PIXELS = 50

# 追蹤記憶。數字越大，短暫漏偵時越容易接回原本球員。
TRACK_MEMORY_FRAMES = 90
POSITION_MATCH_PIXELS = 280

# 補偵測：主 YOLO 漏掉球員時，裁切上一幀位置附近再跑一次 YOLO。
ENABLE_CROP_RECOVERY = True
CROP_CONFIDENCE = 0.08
CROP_IMAGE_SIZE = 960
CROP_SCALE = 2.8
MIN_CROP_SIZE = 420

KEYPOINT_DRAW_CONFIDENCE = 0.20
COURT_CENTER_Y = float(COURT_POLYGON[:, 1].mean())
PLAYER_LABELS = {
    "far": "Player 1",
    "near": "Player 2",
}
COCO_POSE_PAIRS = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

KEYPOINT_COUNT = 17


def output_mode_flags() -> tuple[bool, bool]:
    if OUTPUT_MODE == "video":
        return True, False
    if OUTPUT_MODE == "csv":
        return False, True
    if OUTPUT_MODE == "both":
        return True, True
    raise ValueError('OUTPUT_MODE must be "video", "csv", or "both"')


def skeleton_csv_fieldnames() -> list[str]:
    fieldnames = [
        "video_name",
        "frame_index",
        "player_side",
        "track_id",
        "recovered",
        "detected_side",
        "score",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "point_x",
        "point_y",
        "court_distance",
    ]

    for index in range(1, KEYPOINT_COUNT + 1):
        fieldnames.extend(
            [
                f"keypoint_{index}_x",
                f"keypoint_{index}_y",
                f"keypoint_{index}_conf",
            ]
        )

    return fieldnames


def player_to_csv_row(video_name: str, frame_index: int, player: dict) -> dict:
    box = player["box"]
    keypoints = player.get("keypoints")
    row = {
        "video_name": video_name,
        "frame_index": frame_index,
        "player_side": player["side"],
        "track_id": player.get("track_id"),
        "recovered": bool(player.get("recovered", False)),
        "detected_side": player.get("detected_side"),
        "score": player.get("score"),
        "box_x1": float(box[0]),
        "box_y1": float(box[1]),
        "box_x2": float(box[2]),
        "box_y2": float(box[3]),
        "point_x": float(player["point"][0]),
        "point_y": float(player["point"][1]),
        "court_distance": float(player.get("court_distance", 0.0)),
    }

    if keypoints is None:
        for index in range(1, KEYPOINT_COUNT + 1):
            row[f"keypoint_{index}_x"] = None
            row[f"keypoint_{index}_y"] = None
            row[f"keypoint_{index}_conf"] = None
        return row

    for index, (x, y, confidence) in enumerate(keypoints, start=1):
        row[f"keypoint_{index}_x"] = float(x)
        row[f"keypoint_{index}_y"] = float(y)
        row[f"keypoint_{index}_conf"] = float(confidence)

    return row


def detection_point(box_xyxy: np.ndarray, keypoints_xy_conf: np.ndarray | None) -> tuple[float, float]:
    """Use ankle midpoint when visible; otherwise use the bottom-center of the box."""
    if keypoints_xy_conf is not None:
        visible_ankles = []
        for index in (15, 16):  # COCO left/right ankle
            x, y, conf = keypoints_xy_conf[index]
            if conf > 0.25 and x > 0 and y > 0:
                visible_ankles.append((float(x), float(y)))

        if visible_ankles:
            return tuple(np.mean(visible_ankles, axis=0))

    x1, _, x2, y2 = box_xyxy
    return float((x1 + x2) / 2), float(y2)


def court_polygon_for_cv(court_polygon: np.ndarray | None = None) -> np.ndarray:
    """Convert TL, TR, BL, BR into OpenCV contour order: TL, TR, BR, BL."""
    polygon = COURT_POLYGON if court_polygon is None else court_polygon
    return polygon[[0, 1, 3, 2]].astype(np.int32)


def set_court_polygon(court_points: np.ndarray, source: str) -> None:
    global COURT_POLYGON, COURT_CENTER_Y, MANUAL_COURT_POLYGON

    polygon = np.asarray(court_points, dtype=np.int32)
    COURT_POLYGON = polygon
    MANUAL_COURT_POLYGON = COURT_POLYGON.copy()
    COURT_CENTER_Y = float(COURT_POLYGON[:, 1].mean())
    print(f"court polygon loaded from {source}: {COURT_POLYGON.tolist()}")


def court_select_csv_for_video(video_source: Path) -> Path:
    if RUN_MODE == "single":
        return COURT_SELECT_OUTPUT_PATH
    return OUTPUT_DIR / f"{video_source.stem}_court.csv"


def run_court_select(video_source: Path, court_csv_path: Path) -> None:
    court_csv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(COURT_SELECT_SCRIPT),
        str(video_source),
        str(court_csv_path),
    ]
    print(f"running court_select: {' '.join(command)}")
    subprocess.run(command, check=True)


def setup_court_polygon_with_court_select() -> None:
    if not USE_COURT_SELECT:
        return

    if RUN_MODE == "single":
        video_source = VIDEO_SOURCE
    elif RUN_MODE == "folder":
        videos = video_files(INPUT_DIR)
        if not videos:
            print(f"court_select skipped: no videos found in {INPUT_DIR}")
            return
        video_source = videos[0]
    else:
        return

    court_csv_path = court_select_csv_for_video(video_source)
    if FORCE_COURT_SELECT or not court_csv_path.exists():
        run_court_select(video_source, court_csv_path)
    else:
        print(f"using existing court_select csv: {court_csv_path}")

    court_polygon = load_outer_corners_from_csv(court_csv_path)
    set_court_polygon(court_polygon, str(court_csv_path))


def point_inside_court(point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(court_polygon_for_cv(), point, False) >= 0


def distance_from_court(point: tuple[float, float]) -> float:
    """Positive means inside court, negative means outside court."""
    return cv2.pointPolygonTest(court_polygon_for_cv(), point, True)


def point_near_court(point: tuple[float, float]) -> bool:
    x, y = point
    min_x, min_y = COURT_POLYGON.min(axis=0)
    max_x, max_y = COURT_POLYGON.max(axis=0)

    if y < min_y - COURT_TOP_MARGIN_PIXELS:
        return False
    if y > max_y + COURT_BOTTOM_MARGIN_PIXELS:
        return False
    if x < min_x - COURT_SIDE_MARGIN_PIXELS or x > max_x + COURT_SIDE_MARGIN_PIXELS:
        return False

    return distance_from_court(point) >= -COURT_MARGIN_PIXELS


def point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(first) - np.array(second)))


def player_score(
    box_xyxy: np.ndarray,
    point: tuple[float, float],
    court_distance: float,
    recently_selected: bool,
) -> float:
    """Prefer tracked players, larger detections, and people closest to the court."""
    x1, y1, x2, y2 = box_xyxy
    area = max(0.0, float((x2 - x1) * (y2 - y1)))
    court_center = COURT_POLYGON.mean(axis=0)
    center_distance = np.linalg.norm(np.array(point) - court_center)
    outside_penalty = max(0.0, -court_distance)
    track_bonus = 100000.0 if recently_selected else 0.0
    return float(area + track_bonus - center_distance * 20.0 - outside_penalty * 80.0)


def court_side(point: tuple[float, float]) -> str:
    return "far" if point[1] < COURT_CENTER_Y else "near"


def draw_player_label(image: np.ndarray, box_xyxy: np.ndarray, label: str) -> None:
    x1, y1, _, _ = box_xyxy.astype(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    padding = 6

    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    label_x = max(0, x1)
    label_y = max(text_h + padding * 2, y1 - 8)

    cv2.rectangle(
        image,
        (label_x, label_y - text_h - padding * 2),
        (label_x + text_w + padding * 2, label_y + baseline),
        (0, 120, 255),
        -1,
    )
    cv2.putText(
        image,
        label,
        (label_x + padding, label_y - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_pose(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    keypoints_xy_conf: np.ndarray,
    box_color: tuple[int, int, int] = (0, 180, 255),
) -> None:
    x1, y1, x2, y2 = box_xyxy.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    for first, second in COCO_POSE_PAIRS:
        x_first, y_first, conf_first = keypoints_xy_conf[first]
        x_second, y_second, conf_second = keypoints_xy_conf[second]
        if conf_first >= KEYPOINT_DRAW_CONFIDENCE and conf_second >= KEYPOINT_DRAW_CONFIDENCE:
            cv2.line(
                image,
                (int(x_first), int(y_first)),
                (int(x_second), int(y_second)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    for x, y, confidence in keypoints_xy_conf:
        if confidence >= KEYPOINT_DRAW_CONFIDENCE and x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), 4, (0, 120, 255), -1, cv2.LINE_AA)


class PlayerSelector:
    def __init__(self) -> None:
        self.frame_index = 0
        self.recent_tracks: dict[int, dict] = {}
        self.players_by_side: dict[str, dict] = {}

    def selected_players(self, result) -> list[dict]:
        self.frame_index += 1

        if result.boxes is None or len(result.boxes) == 0:
            self._forget_old_tracks()
            return []

        boxes_raw: Any = result.boxes.xyxy
        boxes = boxes_raw.cpu().numpy()
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        keypoints = None
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints_raw: Any = result.keypoints.data
            keypoints = keypoints_raw.cpu().numpy()

        candidates = []
        for index, box in enumerate(boxes):
            kpts = keypoints[index] if keypoints is not None else None
            point = detection_point(box, kpts)
            court_distance = distance_from_court(point)
            track_id = int(track_ids[index]) if track_ids is not None else None
            recent_track = self.recent_tracks.get(track_id) if track_id is not None else None
            recently_selected = (
                recent_track is not None
                and self.frame_index - recent_track["frame"] <= TRACK_MEMORY_FRAMES
            )
            near_court = point_near_court(point)
            matched_recent_player = near_court and self._matches_recent_player(point)

            if near_court or recently_selected or matched_recent_player:
                detected_side = court_side(point)
                side = detected_side
                if recently_selected and recent_track is not None:
                    side = recent_track["side"]
                candidates.append(
                    {
                        "base_score": player_score(box, point, court_distance, recently_selected),
                        "index": index,
                        "track_id": track_id,
                        "box": box,
                        "point": point,
                        "keypoints": kpts,
                        "side": side,
                        "detected_side": detected_side,
                        "court_distance": court_distance,
                    }
                )

        selected = self._best_assignment(candidates)

        self._remember_tracks(selected[:MAX_PLAYERS])
        return selected[:MAX_PLAYERS]

    def recent_player_state(self, side: str) -> dict | None:
        player_state = self.players_by_side.get(side)
        if player_state is None:
            return None

        if self.frame_index - player_state["frame"] > TRACK_MEMORY_FRAMES:
            return None

        return player_state

    def remember_recovered_player(self, player: dict) -> None:
        self._remember_tracks([player])

    def _best_assignment(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        candidate_indexes = list(range(len(candidates)))
        options = [None, *candidate_indexes]
        best_score = float("-inf")
        best_pair = (None, None)

        for far_index in options:
            for near_index in options:
                if far_index is None and near_index is None:
                    continue
                if far_index is not None and far_index == near_index:
                    continue

                score = 0.0
                if far_index is not None:
                    score += self._score_for_side(candidates[far_index], "far")
                if near_index is not None:
                    score += self._score_for_side(candidates[near_index], "near")

                if score > best_score:
                    best_score = score
                    best_pair = (far_index, near_index)

        selected = []
        for side, candidate_index in zip(("far", "near"), best_pair):
            if candidate_index is None:
                continue
            candidate = candidates[candidate_index]
            candidate["side"] = side
            candidate["score"] = self._score_for_side(candidate, side)
            selected.append(candidate)

        return selected

    def _score_for_side(self, candidate: dict, side: str) -> float:
        score = candidate["base_score"]
        track_id = candidate["track_id"]

        if candidate["detected_side"] == side:
            score += 80000.0
        else:
            score -= 80000.0

        recent_track = self.recent_tracks.get(track_id) if track_id is not None else None
        if recent_track is not None and recent_track["side"] == side:
            score += 350000.0

        player_state = self.players_by_side.get(side)
        if player_state is not None:
            age = self.frame_index - player_state["frame"]
            if age <= TRACK_MEMORY_FRAMES:
                distance = point_distance(candidate["point"], player_state["point"])
                score += max(0.0, POSITION_MATCH_PIXELS - distance) * 1800.0
                if distance > POSITION_MATCH_PIXELS * 1.8:
                    score -= 650000.0 + distance * 500.0

                if track_id is not None and track_id == player_state.get("track_id"):
                    score += 450000.0

        return score

    def _matches_recent_player(self, point: tuple[float, float]) -> bool:
        for player_state in self.players_by_side.values():
            age = self.frame_index - player_state["frame"]
            if age <= TRACK_MEMORY_FRAMES and point_distance(point, player_state["point"]) <= POSITION_MATCH_PIXELS:
                return True
        return False

    def _remember_tracks(self, players: list[dict]) -> None:
        for player in players:
            self.players_by_side[player["side"]] = {
                "frame": self.frame_index,
                "track_id": player["track_id"],
                "point": player["point"],
                "box": player["box"],
            }

            track_id = player["track_id"]
            if track_id is not None:
                self.recent_tracks[track_id] = {
                    "frame": self.frame_index,
                    "side": player["side"],
                }
        self._forget_old_tracks()

    def _forget_old_tracks(self) -> None:
        stale_track_ids = [
            track_id
            for track_id, track in self.recent_tracks.items()
            if self.frame_index - track["frame"] > TRACK_MEMORY_FRAMES
        ]
        for track_id in stale_track_ids:
            del self.recent_tracks[track_id]

        stale_sides = [
            side
            for side, player_state in self.players_by_side.items()
            if self.frame_index - player_state["frame"] > TRACK_MEMORY_FRAMES
        ]
        for side in stale_sides:
            del self.players_by_side[side]


def crop_bounds_from_player_state(
    player_state: dict,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    box = player_state.get("box")
    if box is not None:
        x1, y1, x2, y2 = box
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)
        crop_width = max(float(x2 - x1) * CROP_SCALE, MIN_CROP_SIZE)
        crop_height = max(float(y2 - y1) * CROP_SCALE, MIN_CROP_SIZE)
    else:
        center_x, center_y = player_state["point"]
        crop_width = MIN_CROP_SIZE
        crop_height = MIN_CROP_SIZE

    x1 = int(max(0, center_x - crop_width / 2))
    y1 = int(max(0, center_y - crop_height / 2))
    x2 = int(min(frame_width, center_x + crop_width / 2))
    y2 = int(min(frame_height, center_y + crop_height / 2))
    return x1, y1, x2, y2


def recover_player_from_crop(
    model: YOLO,
    frame: np.ndarray,
    player_selector: PlayerSelector,
    side: str,
) -> dict | None:
    player_state = player_selector.recent_player_state(side)
    if player_state is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds_from_player_state(player_state, frame_width, frame_height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_results = model.predict(
        source=crop,
        conf=CROP_CONFIDENCE,
        imgsz=CROP_IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )
    if not crop_results:
        return None

    result = crop_results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes_raw: Any = result.boxes.xyxy
    boxes = boxes_raw.cpu().numpy()
    keypoints = None
    if result.keypoints is not None and result.keypoints.data is not None:
        keypoints_raw: Any = result.keypoints.data
        keypoints = keypoints_raw.cpu().numpy()

    best_candidate = None
    best_score = float("-inf")
    expected_point = player_state["point"]

    for index, box in enumerate(boxes):
        kpts = keypoints[index].copy() if keypoints is not None else None
        full_box = box.copy()
        full_box[[0, 2]] += crop_x1
        full_box[[1, 3]] += crop_y1

        if kpts is not None:
            kpts[:, 0] += crop_x1
            kpts[:, 1] += crop_y1

        point = detection_point(full_box, kpts)
        distance = point_distance(point, expected_point)
        court_distance = distance_from_court(point)
        score = player_score(full_box, point, court_distance, recently_selected=True) - distance * 1200.0

        if point_near_court(point) and score > best_score:
            best_score = score
            best_candidate = {
                "score": score,
                "index": None,
                "track_id": player_state.get("track_id"),
                "box": full_box,
                "point": point,
                "side": side,
                "keypoints": kpts,
                "court_distance": court_distance,
                "recovered": True,
            }

    return best_candidate


def recover_missing_players(
    model: YOLO,
    frame: np.ndarray,
    player_selector: PlayerSelector,
    players: list[dict],
) -> list[dict]:
    if not ENABLE_CROP_RECOVERY:
        return players

    selected_sides = {player["side"] for player in players}
    recovered_players = []

    for side in ("far", "near"):
        if side in selected_sides:
            continue

        recovered_player = recover_player_from_crop(model, frame, player_selector, side)
        if recovered_player is not None:
            player_selector.remember_recovered_player(recovered_player)
            recovered_players.append(recovered_player)

    return [*players, *recovered_players][:MAX_PLAYERS]


def output_path_for_video(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_path.stem}_player_only.mp4"


def csv_path_for_video(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_path.stem}_player_only.csv"


def output_paths_for_video(video_path: Path) -> tuple[Path | None, Path | None]:
    save_video, save_csv = output_mode_flags()
    video_output = output_path_for_video(video_path) if save_video else None
    csv_output = csv_path_for_video(video_path) if save_csv else None
    if RUN_MODE == "single":
        if video_output is not None:
            video_output = OUTPUT_PATH
        if csv_output is not None:
            csv_output = OUTPUT_PATH.with_suffix(".csv")
    return video_output, csv_output


def video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def print_progress(video_name: str, current_frame: int, total_frames: int) -> None:
    if total_frames <= 0:
        sys.stdout.write(f"\r{video_name}: processed {current_frame} frames")
        sys.stdout.flush()
        return

    progress = min(1.0, current_frame / total_frames)
    filled_width = int(PROGRESS_BAR_WIDTH * progress)
    bar = "#" * filled_width + "-" * (PROGRESS_BAR_WIDTH - filled_width)
    percent = progress * 100
    sys.stdout.write(
        f"\r{video_name}: [{bar}] {percent:6.2f}% "
        f"({current_frame}/{total_frames} frames)"
    )
    sys.stdout.flush()


def process_video(
    model: YOLO,
    crop_model: YOLO,
    video_source: Path,
    output_video_path: Path | None,
    output_csv_path: Path | None,
) -> int:
    source_capture = cv2.VideoCapture(str(video_source))
    fps = source_capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_capture.release()

    if width <= 0 or height <= 0:
        print(f"skipped unreadable video: {video_source}")
        return 0

    save_video = output_video_path is not None
    save_csv = output_csv_path is not None

    writer = None
    if save_video:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (width, height),
        )

    csv_file = None
    csv_writer = None
    if save_csv:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_csv_path.open("w", newline="", encoding="utf-8-sig")
        csv_writer = csv.DictWriter(csv_file, fieldnames=skeleton_csv_fieldnames())
        csv_writer.writeheader()

    results = model.track(
        source=str(video_source),
        stream=True,
        persist=False,
        tracker=TRACKER,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )

    player_selector = PlayerSelector()
    recovered_frame_count = 0
    processed_frame_count = 0

    try:
        for processed_frame_count, result in enumerate(results, start=1):
            players = player_selector.selected_players(result)
            players = recover_missing_players(crop_model, result.orig_img, player_selector, players)
            keep = [player["index"] for player in players if player["index"] is not None]

            if save_video:
                annotated = result[keep].plot(labels=False, conf=False) if keep else result.orig_img.copy()

                for player in players:
                    if player.get("recovered") and player.get("keypoints") is not None:
                        recovered_frame_count += 1
                        draw_pose(annotated, player["box"], player["keypoints"])
                    draw_player_label(annotated, player["box"], PLAYER_LABELS[player["side"]])

                if DRAW_COURT_ROI:
                    cv2.polylines(annotated, [court_polygon_for_cv()], True, (0, 255, 255), 2)

                video_writer: Any = writer
                assert video_writer is not None
                video_writer.write(annotated)

            else:
                for player in players:
                    if player.get("recovered") and player.get("keypoints") is not None:
                        recovered_frame_count += 1

            if save_csv and csv_writer is not None:
                for player in players:
                    csv_writer.writerow(player_to_csv_row(video_source.name, processed_frame_count, player))

            should_update_progress = (
                processed_frame_count == 1
                or processed_frame_count % PROGRESS_UPDATE_FRAMES == 0
                or processed_frame_count == total_frames
            )
            if should_update_progress:
                print_progress(video_source.name, processed_frame_count, total_frames)

        if processed_frame_count:
            print_progress(video_source.name, processed_frame_count, total_frames)
            print()
    finally:
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    if save_video:
        print(f"saved video: {output_video_path}")
    if save_csv:
        print(f"saved csv: {output_csv_path}")
    print(f"crop recovery used: {recovered_frame_count} player frames")
    return recovered_frame_count


def process_single_video(model: YOLO, crop_model: YOLO) -> None:
    print(f"processing single video: {VIDEO_SOURCE}")
    output_video_path, output_csv_path = output_paths_for_video(VIDEO_SOURCE)
    recovered_count = process_video(model, crop_model, VIDEO_SOURCE, output_video_path, output_csv_path)
    if output_video_path is not None:
        print(f"done: saved to {output_video_path}")
    if output_csv_path is not None:
        print(f"done: saved to {output_csv_path}")
    print(f"total crop recovery used: {recovered_count} player frames")


def process_video_folder(model: YOLO, crop_model: YOLO) -> None:
    videos = video_files(INPUT_DIR)
    if not videos:
        print(f"no videos found in: {INPUT_DIR}")
        return

    print(f"found {len(videos)} videos in {INPUT_DIR}")
    total_recovered = 0

    for index, video_source in enumerate(videos, start=1):
        output_video_path, output_csv_path = output_paths_for_video(video_source)
        print(f"\nfolder progress: {index}/{len(videos)} videos")
        print(f"processing: {video_source.name}")
        total_recovered += process_video(
            model,
            crop_model,
            video_source,
            output_video_path,
            output_csv_path,
        )

    if output_mode_flags()[0]:
        print(f"done: {len(videos)} videos saved to {OUTPUT_DIR}")
    if output_mode_flags()[1]:
        print(f"done: {len(videos)} csv files saved to {OUTPUT_DIR}")
    print(f"total crop recovery used: {total_recovered} player frames")


def validate_settings() -> None:
    if RUN_MODE not in {"single", "folder"}:
        raise ValueError('RUN_MODE must be "single" or "folder"')

    if OUTPUT_MODE not in {"video", "csv", "both"}:
        raise ValueError('OUTPUT_MODE must be "video", "csv", or "both"')

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    if RUN_MODE == "single" and not VIDEO_SOURCE.exists():
        raise FileNotFoundError(f"VIDEO_SOURCE not found: {VIDEO_SOURCE}")

    if RUN_MODE == "folder" and not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR}")

    if USE_COURT_SELECT and not COURT_SELECT_SCRIPT.exists():
        raise FileNotFoundError(f"COURT_SELECT_SCRIPT not found: {COURT_SELECT_SCRIPT}")


def load_yolo_models() -> tuple[YOLO, YOLO]:
    print(f"loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    crop_model = YOLO(MODEL_PATH)
    return model, crop_model


def run_selected_mode(model: YOLO, crop_model: YOLO) -> None:
    if RUN_MODE == "single":
        process_single_video(model, crop_model)
    elif RUN_MODE == "folder":
        process_video_folder(model, crop_model)
    else:
        raise ValueError('RUN_MODE must be "single" or "folder"')


def main() -> None:
    validate_settings()
    setup_court_polygon_with_court_select()
    model, crop_model = load_yolo_models()
    run_selected_mode(model, crop_model)


if __name__ == "__main__":
    main()
