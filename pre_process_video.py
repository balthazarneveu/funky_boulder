from pathlib import Path
from typing import Optional
import logging
import cv2
from interactive_pipe.data_objects.image import Image
from properties import FRAMES, THUMBS, FRAME_IDX, TS, FOLDER, PATH_LIST, SIZE


def save_video_frames(input_path: Path, output_folder: Path, trim=None, resize: Optional[float] = None):
    video_name = input_path.stem
    video = cv2.VideoCapture(str(input_path))
    total_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    start, end = None, None

    if trim is not None:
        assert len(trim) == 2
        start, end = trim
        if start is not None:
            start = int(start*total_length)
        if end is not None:
            end = int(end*total_length)
    if (video.isOpened() == False):
        logging.warning("Error opening video stream or file")
    frame_idx = -1
    pth_list = []
    frame_indices = []
    frame_ts = []
    while (video.isOpened()):
        # Capture frame-by-frame
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
        pth = output_folder/f'{video_name}_{frame_idx:05d}.jpg'
        frame_indices.append(frame_idx)
        frame_ts.append(frame_idx/fps)
        pth_list.append(str(pth))
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


def preprocess(input_video_path: Path, output_folder: Path) -> None:
    assert input_video_path.exists()
    output_folder = output_folder/input_video_path.stem
    output_folder.mkdir(exist_ok=True, parents=True)
    save_video_frames(input_video_path, output_folder)


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
    args = parser.parse_args()
    video_path = Path(args.video)
    output_folder = Path(args.output_folder)
    preprocess(video_path, output_folder)
