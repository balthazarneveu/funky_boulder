import numpy as np
import cv2
from interactive_pipe.helper import _private
from interactive_pipe import interactive, interactive_pipeline, Image
from interactive_pipe.data_objects.parameters import Parameters
import argparse
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from pathlib import Path
_private.registered_controls_names = []
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray

SELECTED_BODY_PART = {
    "right_hand": {"mp_index": mp_pose.PoseLandmark.RIGHT_THUMB, "color": (255, 255, 0)},
    "left_hand": {"mp_index": mp_pose.PoseLandmark.LEFT_THUMB, "color": (255, 0, 255)},
    "right_foot": {"mp_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, "color": (0, 255, 255)},
    "left_foot": {"mp_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX, "color": (0, 255, 0)},
}


def select_frames(frame: int = 0, context: dict = {}) -> None:
    context["frame"] = frame


def load_all_poses(video_frames: list[Path], context: dict = {}) -> dict:
    pose_results = {}
    for idx, frame in enumerate(video_frames):
        pose_results[idx] = Parameters.load_yaml(frame.with_suffix(".yaml"))
    context["pose_results"] = pose_results


def select_scale(scale: float = 0.25, context: dict = {}) -> None:
    context["scale"] = scale


def load_frame(video_frames: list[Path], context: dict = {}) -> np.ndarray:
    idx = context["frame"]
    return Image.from_file(video_frames[idx]).data


def resize_frame(frame, context: dict = {}) -> np.ndarray:
    scale = context["scale"]
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)


def load_pose_result(context: dict = {}) -> dict:
    return context["pose_results"][context["frame"]]


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


def get_specific_body_tracks(context: dict = {}):
    context["track"] = {}
    all_pose_results = context["pose_results"]
    for selected_body_part, landmark_info in SELECTED_BODY_PART.items():
        all_coordinates = []
        landmark_index = landmark_info["mp_index"]

        x, y = 0, 0
        # logging.debug(f"Tracking {selected_body_part}, index: {landmark_index}, color: {landmark_color}")

        for timestep, pose_result in all_pose_results.items():
            if "landmarks" in pose_result:
                selected_element = pose_result["landmarks"][landmark_index]
                # x, y = int(selected_element["x"]*frame.shape[1]
                #            ), int(selected_element["y"]*frame.shape[0])
                x, y = selected_element["x"], selected_element["y"]
            else:
                pass
            all_coordinates.append((x, y))
        context["track"][selected_body_part] = all_coordinates


def highlight_tracks(frame, context: dict = {}):
    annot_frame = frame.copy()
    for selected_body_part, landmark_info in SELECTED_BODY_PART.items():
        landmark_color = landmark_info["color"]
        all_coordinates = context["track"][selected_body_part]
        window_size = 5
        current_idx = context["frame"]
        start_idx = max(0, current_idx-window_size)
        end_idx = min(len(all_coordinates), current_idx)
        selected_coordinates = all_coordinates[start_idx:end_idx]
        for idx, (x, y) in enumerate(selected_coordinates):
            cv2.circle(annot_frame, (int(
                x*frame.shape[1]), int(y*frame.shape[0])), 8, landmark_color, -1)
    return annot_frame


def process_video(video):
    load_all_poses(video)
    get_specific_body_tracks()
    select_frames()
    frame = load_frame(video)
    pose_result = load_pose_result()
    frame_overlay = overlay_pose(frame, pose_result)
    tracked_frame = highlight_tracks(frame)
    select_scale()
    frame = resize_frame(frame)
    frame_overlay = resize_frame(frame_overlay)
    tracked_frame = resize_frame(tracked_frame)
    return frame, frame_overlay, tracked_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed-folder", type=str,
        default="__preprocessed_frames/VID_20241107_150948_tmp"
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
