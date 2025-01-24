import cv2
import os
import argparse

def extract_frames(video_path, output_folder, num_frames=30):
    """
    Extracts a specified number of frames from a video and saves them to an output folder.

    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save the extracted frames.
        num_frames (int): Number of frames to extract. Default is 30.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)  # Interval between frames

    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it matches the interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path} to {output_folder}")

def process_video_folder(input_folder, output_base_folder, num_frames=30):
    """
    Processes all video files in a folder, extracting frames from each.

    Args:
        input_folder (str): Path to the folder containing videos.
        output_base_folder (str): Base folder to save extracted frames.
        num_frames (int): Number of frames to extract per video. Default is 30.
    """
    for video_file in os.listdir(input_folder):
        if video_file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            video_path = os.path.join(input_folder, video_file)
            output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])
            extract_frames(video_path, output_folder, num_frames)

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input-folder", type=str, default="/data/ephemeral/home/sujin/extract_frames_sample/captioner_testdata", help="Path to the folder containing videos.")
    parser.add_argument("--output-folder", type=str, default="/data/ephemeral/home/sujin/extract_30frames/output_frames", help="Path to the folder to save extracted frames.")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to extract per video.")

    args = parser.parse_args()

    process_video_folder(args.input_folder, args.output_folder, args.num_frames)

if __name__ == "__main__":
    main()
