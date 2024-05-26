import os
import cv2
import numpy as np

def run_prediction(video_path: str, 
                   gauge_bbox: list,
                   start_frame_id: int = 0, 
                   window_length: int = 25, 
                   threshold: float = 1, 
                   output_path: str = None):
    
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    assert len(gauge_bbox) == 4, "Invalid gauge bounding box"
    
    video_cap = cv2.VideoCapture(video_path)
    frame_id = start_frame_id
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    x1, y1, x2, y2 = gauge_bbox
    
    if output_path is None:
        output_path = video_path.replace(".mp4", "_output.mp4")
    
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    sliding_window = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_id += 1

        # Extract the region of interest (ROI) based on the bounding box
        roi = frame[y1:y2, x1:x2]

        # Convert the ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Add the ROI to the sliding_window
        sliding_window.append(roi_gray)

        if len(sliding_window) > window_length:
            # Remove the oldest ROI from the window
            sliding_window.pop(0)

        if len(sliding_window) == window_length:
            # Calculate the average ROI in the window
            avg_roi = np.mean(sliding_window, axis=0).astype(np.uint8)

            # Calculate the absolute difference between the average ROI and the current ROI
            diff = cv2.absdiff(avg_roi, roi_gray)

            # Compute the mean of the absolute difference
            mean_diff = np.mean(diff)

            # Check if the mean difference exceeds the threshold
            if mean_diff > threshold:
                label = "moving"
                color = (0, 255, 0)
            else:
                label = "not_moving"
                color = (0, 0, 255)
        else:
            label = "pending"
            color = (255, 0, 0)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the position to place the label below the bounding box
        label_pos = (x1-30, y2 + 30)

        # Draw the label on the frame
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        # Write the frame to the output video
        out.write(frame)

    video_cap.release()
    out.release()