from pathlib import Path
from typing import Optional
import logging
import cv2
from interactive_pipe.data_objects.image import Image
from interactive_pipe.data_objects.parameters import Parameters
from properties import FRAMES, THUMBS, FRAME_IDX, TS, FOLDER, PATH_LIST, SIZE, POSE_RESULTS
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def save_video_frames(
    input_path: Path,
    output_folder: Path,
    trim_ratio=None,
    trim_time=None,
    resize: Optional[float] = None,
    pose=None
) -> dict:
    video_name = input_path.stem
    video = cv2.VideoCapture(str(input_path))
    total_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    start, end = None, None

    if trim_ratio is not None:
        assert trim_time is None
        assert len(trim_ratio) == 2
        start, end = trim_ratio
        if start is not None:
            start = int(start*total_length)
        if end is not None:
            end = int(end*total_length)
    if trim_time is not None:
        assert trim_ratio is None
        assert len(trim_time) == 2
        start, end = trim_time
        if start is not None:
            start = int(fps*start)
        if end is not None:
            end = int(fps*end)
    print(start, end, fps)
    if (video.isOpened() is False):
        logging.warning("Error opening video stream or file")
    frame_idx = -1
    pth_list = []
    frame_indices = []
    frame_ts = []
    results_list = []
    while (video.isOpened()):
        ret, frame = video.read()
        frame_idx += 1
        if not ret:
            break
        if end is not None and frame_idx > end:
            logging.debug(f"LAST FRAME REACHED! {frame_idx}>{end}")
            break
        if start is not None and frame_idx <= start:
            continue
        logging.debug(f"{frame_idx}, {frame.shape}")
        original_size = frame.shape[:2]
        rs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            rs_frame = cv2.resize(rs_frame, None, fx=resize, fy=resize)
            rs_frame_size = frame
        else:
            rs_frame_size = original_size
        pose_estimation_results = pose.process(rs_frame)
        pth = output_folder/f'{video_name}_{frame_idx:05d}.jpg'
        results_path = pth.with_suffix(".yaml")

        if not pose_estimation_results.pose_landmarks:
            pose_estimation_dict = {}
        else:
            landmarks = []
            for landmark in pose_estimation_results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                })
            pose_estimation_dict = {"landmarks": landmarks}
        Parameters(pose_estimation_dict).save(
            results_path)
        frame_indices.append(frame_idx)
        frame_ts.append(frame_idx/fps)
        pth_list.append(str(pth))
        results_list.append(pose_estimation_results)
        Image(rs_frame/255.).save(pth)
    sample_config_file = {
        "start_frame": start,
        "end_frame": end,
        "total_frames": total_length,
        "fps": fps,
        FRAMES: {
            SIZE: original_size,
            FRAME_IDX: frame_indices,
            TS: frame_ts,
            POSE_RESULTS: results_list,
        },
        THUMBS: {
            SIZE: rs_frame_size,
            FRAME_IDX: frame_indices,
            TS: frame_ts,
            FOLDER: str(output_folder),
            PATH_LIST: pth_list
        }
    }
    return sample_config_file


def preprocess(input_video_path: Path, output_folder: Path, trim=None, resize=None) -> None:
    assert input_video_path.exists()
    output_folder = output_folder/input_video_path.stem
    output_folder.mkdir(exist_ok=True, parents=True)
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        save_video_frames(input_video_path, output_folder,
                          pose=pose, trim_time=trim, resize=resize)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str,
        default="__media/VID_20241107_150948.mp4"
    )
    parser.add_argument(
        "--output_folder", type=str,
        default="__preprocessed_frames"
    )
    parser.add_argument(
        "--resize", type=float,
        default=None
    )
    parser.add_argument(
        "--trim", type=float, nargs=2,
        default=None
    )
    args = parser.parse_args()
    video_path = Path(args.video)
    output_folder = Path(args.output_folder)
    preprocess(video_path, output_folder, trim=args.trim, resize=args.resize)
