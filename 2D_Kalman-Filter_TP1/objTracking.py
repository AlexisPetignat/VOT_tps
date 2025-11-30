import cv2
import numpy as np
import Detector
from KalmanFilter import KalmanFilter as KFilter


def main():
    filter = KFilter(0.1, 1, 1, 1, 0.1, 0.1)
    capture = cv2.VideoCapture("./video/randomball.avi")

    trajectory = []  # store estimated centers
    img = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        centers = Detector.detect(frame)
        if len(centers) == 0:
            continue

        # Predict
        pred_x, pred_y = filter.predict()
        predicted_pt = (int(pred_x), int(pred_y))

        # Estimate
        meas_x, meas_y = centers[0]
        est_x, est_y = filter.update(np.array([meas_x, meas_y]))
        estimated_pt = (int(est_x), int(est_y))

        # Save trajectory
        trajectory.append(estimated_pt)

        # Draw detected circle
        cv2.circle(frame, (int(meas_x), int(meas_y)), 10, (0, 255, 0), 2)

        # Draw predicted position (blue rectangle)
        cv2.rectangle(
            frame,
            (predicted_pt[0] - 15, predicted_pt[1] - 15),
            (predicted_pt[0] + 15, predicted_pt[1] + 15),
            (255, 0, 0),
            2,
        )

        # Draw estimated position (red rectangle)
        cv2.rectangle(
            frame,
            (estimated_pt[0] - 15, estimated_pt[1] - 15),
            (estimated_pt[0] + 15, estimated_pt[1] + 15),
            (0, 0, 255),
            2,
        )

        # Draw trajectory in red
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

        # Show result
        # cv2.imshow("Tracking", frame)
        img.append(frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
            break

    # Create video
    if len(img) > 0:
        height, width, layers = img[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            "video.mp4", fourcc=fourcc, fps=30, frameSize=(width, height)
        )

        for frame in img:
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print("Video saved as video.mp4")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
