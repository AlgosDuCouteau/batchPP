from function.vid_get import VideoGet
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)
# Add dictionary to store previous positions and counts
previous_positions = {}
in_count = 0
out_count = 0

counting_regions = [
    {
        "polygon": Polygon([(0, 250), (300, 250), (300, 640), (0, 640)]),  # Polygon points
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

def is_inside_polygon(point, polygon):
    return Point(point).within(polygon)

def main():
    global in_count, out_count
    vid_frame_count = 0

    # Setup Model
    model = YOLO('path-to-model/weights/best.pt')


    # Initialize video capture
    vid_get = VideoGet('path-to-video/video.mp4')
    vid_get.start()
    
    # Get video properties for output
    frame = vid_get.read()
    height, width = frame.shape[:2]
    fps = 60  # You can adjust this value based on your needs
    
    # Initialize video writer
    output_path = 'output1.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        frame = vid_get.read()
        if frame is None:
            break
        vid_frame_count += 1
        # Extract the results
        results = model.track(frame, persist=True, half=True, conf=0.1, iou=0.15, device=0, stream=True, tracker="custom-tracker.yaml")
        for r in results:
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                annotator = Annotator(frame, line_width=2)
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                    
                    # Check current position
                    current_inside = is_inside_polygon(bbox_center, counting_regions[0]["polygon"])
                    
                    # If we have a previous position for this ID
                    if track_id in previous_positions:
                        prev_inside = previous_positions[track_id]
                        
                        # Detect crossing
                        if current_inside and not prev_inside:
                            in_count += 1
                        elif not current_inside and prev_inside:
                            out_count += 1
                    
                    # Update previous position
                    previous_positions[track_id] = current_inside
                    
                    track = track_history[track_id]
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)
                    annotator.box_label(box, str(cls), color=colors(cls, True))

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            
            # Calculate total
            total = in_count - out_count
            
            # Draw the total count in top left
            count_text = f"Total: {total}"
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                count_text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                2
            )
            
            # Position text in top left with padding
            text_x = 20
            text_y = 40
            
            # Draw background rectangle for better visibility
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw the count
            cv2.putText(
                frame, 
                count_text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)

        # # Write the frame to output video
        # out.write(frame)
        
        cv2.imshow("YOLOv8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything
    vid_get.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()