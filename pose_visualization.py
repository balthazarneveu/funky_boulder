import numpy as np
import cv2
from interactive_pipe.helper import _private
from interactive_pipe import interactive, interactive_pipeline, Image
from interactive_pipe.data_objects.parameters import Parameters
import argparse
import mediapipe as mp
from tqdm import tqdm
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
    if "pose_results" not in context:
        pose_results = {}
        for idx, frame in tqdm(
            enumerate(video_frames), total=len(video_frames), desc="Retrieving pose results"
        ):
            pose_results[idx] = Parameters.load_yaml(
                frame.with_suffix(".yaml"))
        context["pose_results"] = pose_results


def select_scale(scale: float = 0.25, context: dict = {}) -> None:
    context["scale"] = scale


def load_frame(video_frames: list[Path], context: dict = {}) -> np.ndarray:
    idx = context["frame"]
    return Image.from_file(video_frames[idx], backend="pillow").data


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
    if "landmarks" in pose_result:
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
    else:
        pass
    annotated_image = annotated_image/255.
    return annotated_image


def get_specific_body_tracks(context: dict = {}):
    if "track" in context:
        return
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


def highlight_tracks(frame, sparkle_size=3, context: dict = {}):
    # Create a copy of the frame for annotations
    background = (frame.copy()*255).astype("uint8")
    # Overlay for transparency effects
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Parameters for visual effects
    window_size = 30
    glow_intensity = 200  # Max intensity for the glow
    sparkle_count = 200  # Number of sparkles per frame

    for selected_body_part, landmark_info in SELECTED_BODY_PART.items():
        landmark_color = np.array(landmark_info["color"])
        all_coordinates = context["track"][selected_body_part]
        current_idx = context["frame"]

        # Determine the trail segment
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(all_coordinates), current_idx)
        if end_idx == start_idx:
            end_idx = start_idx + 1
        selected_coordinates = all_coordinates[start_idx:end_idx]

        # # Draw trails with a glowing effect
        for idx, (x, y) in enumerate(selected_coordinates):
            # Calculate the position and fading effect
            pos = (int(x * frame.shape[1]), int(y * frame.shape[0]))
            # intensity = glow_intensity * (1 - idx / window_size)
            # color = tuple(
            #     (landmark_color * (1 - idx / window_size)).astype(int))
            intensity = glow_intensity * (idx / window_size)
            color = tuple(
                (landmark_color * (idx / window_size)).astype(int))
            # Draw glow (transparent circle with blur effect)
            int_color = tuple(int(c) for c in color)
            cv2.circle(overlay, pos, 20, (*int_color, int(intensity)), -1)

        # Add sparkles near the current position
        x, y = selected_coordinates[-1]
        if sparkle_size > 0:
            for _ in range(sparkle_count):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0.01, 0.05)
                radius_pixels = radius*frame.shape[0]
                sparkle_x = int(
                    x * frame.shape[1] + np.cos(angle) * radius_pixels)
                sparkle_y = int(
                    y * frame.shape[0] + np.sin(angle) * radius_pixels)
                sparkle_color = tuple(np.random.randint(50, 255, 3).tolist())
                cv2.circle(
                    overlay,
                    (sparkle_x, sparkle_y),
                    sparkle_size,
                    (*sparkle_color, 0.1),
                    -1
                )
    alpha = 0.8  # Transparency factor
    blended_frame = cv2.addWeighted(overlay, alpha, background, 1., 0)
    return blended_frame/255.


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
        choices=["qt", "gradio", "mpl", "headless"],
        default="qt"
    )
    parser.add_argument(
        "--trim", type=int, nargs=2,
        default=None,
        help="Trim the video to the specified range of frames"
    )
    parser.add_argument(
        "--speedup", type=float,
        default=1.0,
        help="Speedup factor for the video"
    )
    args = parser.parse_args()
    preprocessed = Path(args.preprocessed_folder)
    frames = sorted(list(preprocessed.glob("*.jpg")))

    gui = args.gui
    live = True
    if gui != "headless":
        interactive(frame=(0, [0, len(frames)-1]))(select_frames)
        interactive(scale=(0.25, [0.1, 2.]))(select_scale)
        interactive(empty_background=(False,))(overlay_pose)
        interactive(sparkle_size=(3, [1, 10]))(highlight_tracks)
        interactive_pipeline(
            gui=gui,
            cache=True,
            safe_input_buffer_deepcopy=False
        )(process_video)(frames)
    else:
        trim = args.trim
        if trim is not None:
            frames = frames[int(trim[0]):int(trim[1])]
        import PIL.Image as PILImage
        img_list = []
        headless_pipeline = interactive_pipeline(
            gui=None,
            safe_input_buffer_deepcopy=False
        )(process_video)
        speedup = args.speedup
        skip_factor = int(np.round(speedup))
        for img_idx in tqdm(range(0, len(frames), skip_factor), desc="Generating frames"):
            # Let's first override some of the default parameters.
            headless_pipeline.parameters = {
                "select_frames": {"frame": img_idx},
                "select_scale": {"scale": 0.25},
                "highlight_tracks": {"sparkle_size": 0},
                "overlay_pose": {
                    "empty_background": False
                }
            }
            headless_pipeline(frames)
            img = (255.*headless_pipeline.results[-1]).astype(np.uint8)
            img_list.append(PILImage.fromarray(img))

        img_list[0].save(f"animation_{preprocessed.stem}_speed={speedup:.1f}x.gif", save_all=True,
                         append_images=img_list[1:], duration=1, loop=1000)
