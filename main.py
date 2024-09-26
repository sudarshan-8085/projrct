import sys
import argparse
import pathlib
import cv2
import utils as util
import time
import os

def initialize_video_sources(video_paths):
    vs = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {path}. Please check the file path.")
            return None
        vs.append(cap)
    return vs

def process_frames(vs):
    frames = []
    for cap in vs:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error in capturing frames.")
            return None
        frames.append(frame)
    return frames

def main(sources):
    # Build video paths
    video_paths = [str(pathlib.Path.cwd() / "datas" / source) for source in sources]
    
    # Initialize video sources
    vs = initialize_video_sources(video_paths)
    if vs is None:
        return

    # Load the YOLOv5 model (update with your actual model file)
    model_path = "C:/Users/sudar/Documents/traffic_monitoring/models/yolov5.weights"  # Change this to your actual model file
    if not os.path.isfile(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        return

    # Load the model
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getUnconnectedOutLayersNames()

    # Initial configuration of lanes
    lanes = util.Lanes([util.Lane("", "", 1), util.Lane("", "", 3), util.Lane("", "", 4), util.Lane("", "", 2)])
    wait_time = 0

    while True:
        frames = process_frames(vs)
        if frames is None:
            break

        # Assign frames to lanes
        for lane in lanes.getLanes():
            lane.frame = frames[lane.lane_number - 1]  # Assuming lane_number is 1-based

        start_time = time.time()
        lanes = util.final_output(net, ln, lanes)
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")

        if wait_time <= 0:
            images_transition = util.display_result(wait_time, lanes)
            final_image = cv2.resize(images_transition, (1020, 720))
            cv2.imshow("Output", final_image)
            cv2.waitKey(100)

            wait_time = util.schedule(lanes)  # Schedule based on lane status

        images_scheduled = util.display_result(wait_time, lanes)
        final_image = cv2.resize(images_scheduled, (1020, 720))
        cv2.imshow("Output", final_image)
        cv2.waitKey(1)
        wait_time -= 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in vs:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine duration based on car count in video feeds.")
    parser.add_argument("--sources", help="Comma-separated video feeds to be inferred on, relative to the 'datas' folder.",
                        type=str, default="video1.mp4,video2.mp4,video3.mp4,video4.mp4")
    args = parser.parse_args()

    sources = args.sources.split(",")
    print(f"Video sources: {sources}")
    main(sources)
