import cv2
import numpy as np
import pandas as pd
from utils import BoundingBox, Point, Track
from scipy.optimize import linear_sum_assignment

pd.options.mode.chained_assignment = None
PATH = "../det/Yolov5l/"
FILE = "det.txt"
FILE_OUTPUT = "../gt/tp3_det.txt"
NAMES = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]

# Tracking thresholds
IOU_THRESHOLD = 0.3  # Minimum IoU to consider a match valid
MAX_MISSED_FRAMES = 20
MIN_OBSERVED_FRAME = 60


def colorize(id: int) -> tuple:
    """Generate a color based on an ID."""
    np.random.seed(id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


def IoU_score(box1: BoundingBox, box2: BoundingBox):
    xA = max(box1.top_left.x, box2.top_left.x)
    yA = max(box1.top_left.y, box2.top_left.y)
    xB = min(box1.bot_right.x, box2.bot_right.x)
    yB = min(box1.bot_right.y, box2.bot_right.y)

    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA) * (yB - yA))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box1.bot_right.x - box1.top_left.x) * (
        box1.bot_right.y - box1.top_left.y
    )
    boxBArea = (box2.bot_right.x - box2.top_left.x) * (
        box2.bot_right.y - box2.top_left.y
    )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def update_frame(
    frame: np.ndarray, box: BoundingBox, box_idx: int, color, inferred=False
) -> np.ndarray:
    # Add rectangles on frame
    cv2.rectangle(
        frame,
        (int(box.top_left.x), int(box.top_left.y)),
        (int(box.bot_right.x), int(box.bot_right.y)),
        color,
        2,
    )
    cv2.putText(
        frame,
        f"ID: {box_idx}" if not inferred else f"ID: {box_idx} (Predicted)",
        (int(box.top_left.x), int(box.top_left.y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    return frame


def main():
    # Load detections
    detections = pd.read_csv(PATH + FILE, header=None, names=NAMES, delimiter=" ")

    # Group detections by frame
    boxes_by_frame = {}
    for frame_num in range(
        int(detections["frame"].min()), int(detections["frame"].max()) + 1
    ):
        df_frame = detections[detections["frame"] == frame_num]
        boxes_by_frame[frame_num] = df_frame.apply(
            lambda row: BoundingBox(
                Point(row["bb_left"], row["bb_top"]),
                Point(
                    row["bb_left"] + row["bb_width"],
                    row["bb_top"] + row["bb_height"],
                ),
            ),
            axis=1,
        ).to_list()

    img = []
    frame_numbers = sorted(boxes_by_frame.keys())

    # Initialize track list with dynamic size
    TRACKS = []
    next_track_id = 0
    df_tracked = detections.copy()

    # Process each frame
    for frame_idx, frame_num in enumerate(frame_numbers):
        # Load frame image
        prefix = "" if frame_num >= 100 else "0" if frame_num >= 10 else "00"
        frame = cv2.imread(f"../img1/000{prefix}{frame_num}.jpg")

        current_detections = boxes_by_frame[frame_num]

        if frame_idx == 0:
            # Initialize tracks with first frame detections
            for box in current_detections:
                track = Track(next_track_id, box)
                TRACKS.append(track)
                next_track_id += 1
                color = colorize(track.track_id)
                frame = update_frame(frame, box, track.track_id, color)

                # Update row in df
                rows = df_tracked.loc[df_tracked["frame"] == frame_num]
                idx = rows.index[rows["bb_left"] == box.top_left.x][0]
                df_tracked.at[idx, "id"] = track.track_id
        else:
            # Get active tracks
            active_tracks = [t for t in TRACKS if t.missed < MAX_MISSED_FRAMES]

            # Mark all tracks as missed initially
            for track in TRACKS:
                track.miss()

            matched_detection_indices = set()

            if len(active_tracks) > 0 and len(current_detections) > 0:
                # Build cost matrix: rows = tracks, cols = detections
                cost_matrix = (
                    np.ones((len(active_tracks), len(current_detections))) * 1e5
                )

                for t_idx, track in enumerate(active_tracks):
                    # Predict the box position using Kalman filter
                    predicted_center = track.predict_kalman()
                    for d_idx, det_box in enumerate(current_detections):
                        # Run hungarian with the predicted position for easier ID matching
                        pred_box = track.box.copy()
                        pred_box.move_center(predicted_center)
                        iou = IoU_score(pred_box, det_box)
                        if iou > IOU_THRESHOLD:
                            cost_matrix[t_idx, d_idx] = 1 - iou

                # Hungarian algorithm
                if cost_matrix.size > 0:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # Process matches
                    for t_idx, d_idx in zip(row_ind, col_ind):
                        # Only accept match if cost is reasonable (IoU > threshold)
                        if cost_matrix[t_idx, d_idx] < 1.0:
                            track = active_tracks[t_idx]
                            det_box = current_detections[d_idx]

                            # Update Kalman filter
                            track.predict_kalman()
                            meas = track.update_kalman(det_box.center())
                            refined_box = det_box.copy()
                            refined_box.move_center(meas)

                            # Update track
                            track.update_box(refined_box)
                            matched_detection_indices.add(d_idx)

                            # Draw on frame
                            color = colorize(track.track_id)
                            frame = update_frame(
                                frame, refined_box, track.track_id, color
                            )

                            # Update row in df
                            rows = df_tracked.loc[df_tracked["frame"] == frame_num]
                            idx = rows.index[rows["bb_left"] == det_box.top_left.x][0]
                            df_tracked.at[idx, "id"] = track.track_id

            # Handle unmatched detections (new tracks)
            for d_idx in range(len(current_detections)):
                if d_idx not in matched_detection_indices:
                    det_box = current_detections[d_idx]
                    new_track = Track(next_track_id, det_box)
                    TRACKS.append(new_track)
                    next_track_id += 1

                    # Draw on frame
                    color = colorize(new_track.track_id)
                    frame = update_frame(frame, det_box, new_track.track_id, color)

                    # Update row in df
                    rows = df_tracked.loc[df_tracked["frame"] == frame_num]
                    idx = rows.index[rows["bb_left"] == det_box.top_left.x][0]
                    df_tracked.at[idx, "id"] = new_track.track_id

            # handle unmatched tracks (missed detections)
            for track in TRACKS:
                if (
                    track.missed > 0
                    and track.missed < MAX_MISSED_FRAMES
                    and track.total_matches >= MIN_OBSERVED_FRAME
                ):
                    # Predict new box
                    predicted_center = track.predict_kalman()
                    pred_box = track.box.copy()
                    pred_box.move_center(predicted_center)

                    # Draw on frame
                    color = colorize(track.track_id)
                    frame = update_frame(
                        frame, pred_box, track.track_id, color, inferred=True
                    )

        img.append(frame)

    # Create video
    if len(img) > 0:
        height, width, layers = img[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            "video.avi", fourcc=fourcc, fps=30, frameSize=(width, height)
        )

        for frame in img:
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print("Video saved as video.avi")

    df_tracked.to_csv(FILE_OUTPUT, header=False, index=False, sep=",")
    print(f"Tracked detections saved as {FILE_OUTPUT}")


if __name__ == "__main__":
    main()
