import numpy as np
import cv2
from interactive_pipe.helper import _private
from interactive_pipe import interactive, interactive_pipeline, Image
from interactive_pipe.data_objects.parameters import Parameters
import argparse
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray


def select_frames(frame: int = 0, context: dict = {}) -> None:
    context["frame"] = frame


def select_scale(scale: float = 0.25, context: dict = {}) -> None:
    context["scale"] = scale


def load_frame(video_frames: list[Path], context: dict = {}) -> np.ndarray:
    idx = context["frame"]
    return Image.from_file(video_frames[idx]).data


def resize_frame(frame, context: dict = {}) -> np.ndarray:
    scale = context["scale"]
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)


def load_pose_result(video_frames: list[Path], context: dict = {}) -> dict:
    idx = context["frame"]
    pose_results = Parameters.load_yaml(video_frames[idx].with_suffix(".yaml"))
    context["pose_results"] = pose_results
    return pose_results


def overlay_pose(frame: np.ndarray, pose_result: dict, empty_background=False) -> np.ndarray:
    if empty_background:
        annotated_image = np.zeros(frame.shape, dtype=np.uint8)
    else:
        annotated_image = (frame.copy() * 255).astype("uint8")
    recreated_landmarks = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(
                x=landmark['x'],
                y=landmark['y'],
                z=landmark['z'],
                visibility=landmark['visibility']
            ) for landmark in pose_result["landmarks"]
        ]
    )
    dw_style = mp_drawing_styles.get_default_pose_landmarks_style()
    mp_drawing.draw_landmarks(
        annotated_image,
        recreated_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=dw_style
    )
    annotated_image = annotated_image/255.
    return annotated_image


def process_video(video):
    select_frames()
    frame = load_frame(video)
    pose_result = load_pose_result(video)
    frame_overlay = overlay_pose(frame, pose_result)
    select_scale()
    frame = resize_frame(frame)
    frame_overlay = resize_frame(frame_overlay)
    # frame_annot = detect_pose(frame)
    return frame, frame_overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--video", type=str,
    #     default="__media/VID_20241107_150948.mp4"
    # )
    parser.add_argument(
        "--preprocessed-folder", type=str,
        default="__preprocessed_frames/VID_20241107_150948"
    )
    parser.add_argument(
        "--gui", type=str,
        choices=["qt", "gradio", "mpl"],
        default="qt"
    )
    args = parser.parse_args()
    preprocessed = Path(args.preprocessed_folder)
    frames = sorted(list(preprocessed.glob("*.jpg")))
    interactive(frame=(0, [0, len(frames)-1]))(select_frames)
    interactive(scale=(0.25, [0.1, 2.]))(select_scale)
    interactive(empty_background=(False,))(overlay_pose)
    gui = args.gui
    interactive_pipeline(
        gui=gui,
        cache=True,
        safe_input_buffer_deepcopy=False
    )(process_video)(frames)
