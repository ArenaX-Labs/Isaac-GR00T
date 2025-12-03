"""
Convert sessions created by SAI Zoo RecordingWrapper to LeRobot dataset format.
"""

import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd

try:
    import imageio

    IMAGEIO_AVAILABLE = True
    CV2_AVAILABLE = False
except ImportError:
    try:
        import cv2

        CV2_AVAILABLE = True
        IMAGEIO_AVAILABLE = False
    except ImportError:
        IMAGEIO_AVAILABLE = False
        CV2_AVAILABLE = False
        print("Warning: Neither imageio nor cv2 available. Videos will not be created.")

STATE_KEYS_MAPPTING = {
    "base": "base",
    "torso": "torso",
    "left_arm": "left_arm",
    "right_arm": "right_arm",
    "left_arm_gripper": "left_hand",
    "right_arm_gripper": "right_hand",
    "right_leg": "right_leg",
    "left_leg": "left_leg",
}


def load_session_metadata(session_dir: str):
    env_metadata_path = os.path.join(session_dir, "env_metadata.npy")
    episode_metadata_path = os.path.join(session_dir, "episode_metadata.npy")

    env_metadata = {}
    episode_metadata = {}

    if os.path.exists(env_metadata_path):
        env_metadata = np.load(env_metadata_path, allow_pickle=True).item()

    if os.path.exists(episode_metadata_path):
        episode_metadata = np.load(episode_metadata_path, allow_pickle=True).item()

    return env_metadata, episode_metadata


def load_episodes_from_hdf5(hdf5_path: str):
    episodes = []

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted([k for k in f.keys() if k.startswith("demo_")])

        if not demo_keys:
            print(f"Warning: No demo episodes found in {hdf5_path}")
            return episodes

        for demo_key in demo_keys:
            demo_grp = f[demo_key]

            try:
                episode_data = {
                    "actions": np.array(demo_grp["actions"]),
                    "rewards": np.array(demo_grp["rewards"]),
                    "proprio": np.array(demo_grp["obs/proprio"]),
                }

                if "timestamps" in demo_grp:
                    episode_data["timestamps"] = np.array(demo_grp["timestamps"])
                else:
                    episode_data["timestamps"] = None

                if "obs/qpos" in demo_grp:
                    qpos_grp = demo_grp["obs/qpos"]
                    episode_data["qpos"] = {}
                    for part_name in qpos_grp.keys():
                        episode_data["qpos"][part_name] = np.array(qpos_grp[part_name])

                if "obs/pixels" in demo_grp:
                    pixels_grp = demo_grp["obs/pixels"]
                    episode_data["pixels"] = {}
                    for cam_name in pixels_grp.keys():
                        episode_data["pixels"][cam_name] = np.array(pixels_grp[cam_name])

                if "info" in demo_grp:
                    info_grp = demo_grp["info"]
                    episode_data["info"] = {}
                    _load_info_group(info_grp, episode_data["info"])

                episodes.append(episode_data)
            except Exception as e:
                print(f"Warning: Failed to load episode {demo_key}: {e}")
                continue

    return episodes


def _load_info_group(info_grp, info_dict):
    for key in info_grp.keys():
        if isinstance(info_grp[key], h5py.Group):
            info_dict[key] = {}
            _load_info_group(info_grp[key], info_dict[key])
        else:
            data = np.array(info_grp[key])
            if data.dtype.kind == "S" or data.dtype.kind == "U":
                data = [s.decode("utf-8") if isinstance(s, bytes) else s for s in data]
            info_dict[key] = data


def build_modality_json(episodes, env_metadata, task_descriptions, action_type="ik"):
    modality = {
        "state": {},
        "action": {},
    }

    if not episodes:
        return modality

    if "qpos" in episodes[0]:
        qpos_keys = sorted(episodes[0]["qpos"].keys())
        qpos_keys = [k for k in qpos_keys if k in STATE_KEYS_MAPPTING]
        state_idx = 0
        action_idx = 0

        for part_name in qpos_keys:
            part_qpos = episodes[0]["qpos"][part_name]
            state_part_dim = part_qpos.shape[1] if len(part_qpos.shape) > 1 else 1

            modality["state"][STATE_KEYS_MAPPTING[part_name]] = {
                "start": state_idx,
                "end": state_idx + state_part_dim,
            }
            if action_type == "joint":
                action_part_dim = part_qpos.shape[1] if len(part_qpos.shape) > 1 else 1
                modality["action"][STATE_KEYS_MAPPTING[part_name]] = {
                    "start": action_idx,
                    "end": action_idx + action_part_dim,
                }
            elif action_type == "ik":
                mapped_name = STATE_KEYS_MAPPTING[part_name]
                if "hand" in mapped_name or mapped_name in ["base", "torso"]:
                    action_part_dim = state_part_dim
                elif "arm" in mapped_name:
                    action_part_dim = 6
                else:
                    action_part_dim = state_part_dim

                modality["action"][mapped_name] = {
                    "start": action_idx,
                    "end": action_idx + action_part_dim,
                }
            else:
                raise ValueError(f"Invalid action type: {action_type}")
            action_idx += action_part_dim
            state_idx += state_part_dim
    else:
        if len(episodes[0]["proprio"].shape) > 0:
            modality["state"]["proprio"] = {
                "start": 0,
                "end": episodes[0]["proprio"].shape[-1],
            }

    if episodes and "pixels" in episodes[0]:
        modality["video"] = {}
        for cam_name in episodes[0]["pixels"].keys():
            modality["video"][cam_name] = {
                "original_key": f"observation.images.{cam_name}",
            }

    if task_descriptions:
        modality["annotation"] = {
            "human.action.task_description": {},
            "human.validity": {},
        }

    return modality


def save_video_from_pixels(pixels_array, output_path, fps=20):
    if not IMAGEIO_AVAILABLE and not CV2_AVAILABLE:
        print(f"Warning: Cannot save video {output_path} - no video library available")
        return False

    try:
        if len(pixels_array.shape) == 4:
            frames = pixels_array
        else:
            frames = [pixels_array]

        if IMAGEIO_AVAILABLE:
            processed_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                processed_frames.append(frame)
            imageio.mimwrite(output_path, processed_frames, fps=fps, codec="libx264")
        elif CV2_AVAILABLE:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in frames:
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()
        return True
    except Exception as e:
        print(f"Warning: Failed to save video {output_path}: {e}")
        return False


def convert_actions_to_joint(episode_data, controller_info, modality):
    """Convert actions to absolute joint positions based on controller type."""
    actions = np.array(episode_data["actions"])

    if "qpos" not in episode_data or not episode_data["qpos"]:
        return actions.tolist()

    qpos_keys = sorted(episode_data["qpos"].keys())
    qpos_keys = [k for k in qpos_keys if k in STATE_KEYS_MAPPTING]

    converted_parts = []

    for part_name in qpos_keys:
        part_qpos = np.array(episode_data["qpos"][part_name])

        part_controller = controller_info.get(part_name, {})
        controller_type = part_controller.get("type", "")
        input_type = part_controller.get("input_type", "delta")

        mapped_part_name = STATE_KEYS_MAPPTING[part_name]

        if mapped_part_name in modality.get("action", {}):
            part_action_start = modality["action"][mapped_part_name]["start"]
            part_action_end = modality["action"][mapped_part_name]["end"]
            part_action_slice = actions[:, part_action_start:part_action_end]
        else:
            raise ValueError(f"{mapped_part_name} not found in modality.action")

        if "OSC" in controller_type or "IK" in controller_type:
            next_qpos = np.roll(part_qpos, -1, axis=0)
            next_qpos[-1] = part_qpos[-1]
            part_actions = next_qpos
        elif controller_type == "JOINT_POSITION":
            if input_type == "absolute":
                part_actions = part_qpos
            elif input_type == "delta":
                if len(part_action_slice.shape) == 1:
                    part_action_slice = part_action_slice.reshape(-1, 1)
                if len(part_qpos.shape) == 1:
                    part_qpos = part_qpos.reshape(-1, 1)
                part_actions = part_qpos + part_action_slice
        else:
            part_actions = part_qpos

        converted_parts.append(part_actions)

    if converted_parts:
        converted_actions = np.concatenate(converted_parts, axis=1)
        return converted_actions.tolist()
    else:
        return actions.tolist()


def create_parquet_data(
    episode_data,
    episode_index,
    task_index,
    global_index,
    modality,
    validity_task_index,
    controller_info=None,
    action_type="ik",
    dt=0.05,
):
    T = len(episode_data["actions"])

    state_list = []
    if "qpos" in episode_data and episode_data["qpos"]:
        qpos_keys = sorted(episode_data["qpos"].keys())
        qpos_keys = [k for k in qpos_keys if k in STATE_KEYS_MAPPTING]
        for t in range(T):
            state_t = []
            for part_name in qpos_keys:
                part_qpos = episode_data["qpos"][part_name][t]
                if np.isscalar(part_qpos):
                    state_t.append(float(part_qpos))
                else:
                    state_t.extend([float(x) for x in part_qpos.flatten()])
            state_list.append(state_t)
    else:
        if len(episode_data["proprio"].shape) == 1:
            state_list = [[float(x)] for x in episode_data["proprio"]]
        else:
            state_list = episode_data["proprio"].tolist()

    if action_type == "joint":
        if controller_info is None:
            controller_info = {}
        action_list = convert_actions_to_joint(episode_data, controller_info, modality)
    else:
        action_list = episode_data["actions"].tolist()

    # Ensure timestamps match video fps (20fps) not wall-clock recording time
    if episode_data.get("timestamps") is not None:
        original_timestamps = episode_data["timestamps"]
        if len(original_timestamps) > 1:
            # Calculate actual recording fps
            recorded_fps = 1.0 / np.mean(np.diff(original_timestamps))
            video_fps = 20.0

            # If recording fps differs significantly from video fps, recalculate
            if abs(recorded_fps - video_fps) > 1.0:
                timestamps = [t / video_fps for t in range(T)]
            else:
                timestamps = original_timestamps.tolist()
        else:
            timestamps = original_timestamps.tolist()
    else:
        timestamps = [t * dt for t in range(T)]

    data = {
        "observation.state": state_list,
        "action": action_list,
        "timestamp": timestamps,
        "annotation.human.action.task_description": [task_index] * T,
        "task_index": [task_index] * T,
        "annotation.human.validity": [validity_task_index] * T,
        "episode_index": [episode_index] * T,
        "index": list(range(global_index, global_index + T)),
        "next.reward": list(episode_data["rewards"]),
        "next.done": [False] * (T - 1) + [True],
    }

    df = pd.DataFrame(data)
    return df


def convert_sessions_to_lerobot(input_dir, output_dir, action_type="ik"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)

    session_dirs = []
    for root, dirs, files in os.walk(input_dir):
        hdf5_path = os.path.join(root, "episodes.hdf5")
        if os.path.exists(hdf5_path):
            session_dirs.append(root)

    session_dirs.sort()

    if not session_dirs:
        raise ValueError(f"No session directories found in {input_dir}")

    print(f"Found {len(session_dirs)} session directories")

    all_episodes = []
    all_env_metadata = []
    task_descriptions = []

    for session_dir in session_dirs:
        print(f"Processing session: {session_dir}")

        env_metadata, episode_metadata = load_session_metadata(session_dir)
        all_env_metadata.append(env_metadata)

        hdf5_path = os.path.join(session_dir, "episodes.hdf5")
        episodes = load_episodes_from_hdf5(hdf5_path)

        for episode in episodes:
            all_episodes.append(episode)
            task_descriptions.append(
                episode["info"]["task_desc"][0].decode()
            )  # Assume only one task description per episode

    unique_tasks = sorted(list(set(task_descriptions)))
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

    modality = build_modality_json(
        all_episodes,
        all_env_metadata[0] if all_env_metadata else {},
        unique_tasks,
        action_type,
    )

    with open(os.path.join(output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)

    with open(os.path.join(output_dir, "meta", "tasks.jsonl"), "w") as f:
        task_idx = 0
        for task in unique_tasks:
            f.write(json.dumps({"task_index": task_idx, "task": task}) + "\n")
            task_idx += 1

    global_index = 0
    episodes_jsonl = []

    validity_task_index = len(unique_tasks)

    for episode_idx, (episode_data, task_desc) in enumerate(zip(all_episodes, task_descriptions)):
        task_index = task_to_index[task_desc]
        T = len(episode_data["actions"])

        controller_info = all_env_metadata[0].get("controller_info", {}) if all_env_metadata else {}

        df = create_parquet_data(
            episode_data,
            episode_idx,
            task_index,
            global_index,
            modality,
            validity_task_index,
            controller_info=controller_info,
            action_type=action_type,
        )

        parquet_path = os.path.join(output_dir, "data", "chunk-000", f"episode_{episode_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)

        if "pixels" in episode_data and episode_data["pixels"]:
            videos_dir = os.path.join(output_dir, "videos", "chunk-000")
            os.makedirs(videos_dir, exist_ok=True)

            for cam_name, pixels_array in episode_data["pixels"].items():
                cam_dir = os.path.join(videos_dir, f"observation.images.{cam_name}")
                os.makedirs(cam_dir, exist_ok=True)

                video_path = os.path.join(cam_dir, f"episode_{episode_idx:06d}.mp4")
                save_video_from_pixels(pixels_array, video_path, fps=20)

        episodes_jsonl.append(
            {
                "episode_index": episode_idx,
                "tasks": [task_desc],
                "length": T,
            }
        )

        global_index += T

    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep_info in episodes_jsonl:
            f.write(json.dumps(ep_info) + "\n")

    total_timesteps = sum(ep["length"] for ep in episodes_jsonl)

    # Build features section
    features = {}

    # Add video features
    if all_episodes and "pixels" in all_episodes[0] and all_episodes[0]["pixels"]:
        for cam_name, pixels_array in all_episodes[0]["pixels"].items():
            if len(pixels_array.shape) >= 3:
                # Shape is typically (T, H, W, C) or (T, H, W)
                if len(pixels_array.shape) == 4:
                    height, width, channels = (
                        pixels_array.shape[1],
                        pixels_array.shape[2],
                        pixels_array.shape[3],
                    )
                elif len(pixels_array.shape) == 3:
                    height, width = pixels_array.shape[1], pixels_array.shape[2]
                    channels = 1
                else:
                    continue
                video_key = f"observation.images.{cam_name}"
                features[video_key] = {
                    "dtype": "video",
                    "shape": [int(height), int(width), int(channels)],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 20.0,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }

    if all_episodes and len(all_episodes[0]["actions"]) > 0:
        first_episode = all_episodes[0]
        if "qpos" in first_episode and first_episode["qpos"]:
            qpos_keys = sorted(first_episode["qpos"].keys())
            qpos_keys = [k for k in qpos_keys if k in STATE_KEYS_MAPPTING]
            state_dim = sum(
                part_qpos.shape[1] if len(part_qpos.shape) > 1 else 1
                for part_name, part_qpos in first_episode["qpos"].items()
                if part_name in qpos_keys and part_qpos.shape[0] > 0
            )
        else:
            state_dim = first_episode["proprio"].shape[-1] if len(first_episode["proprio"].shape) > 0 else 1

        features["observation.state"] = {
            "dtype": "float64",
            "shape": [int(state_dim)],
        }

        if modality.get("state"):
            state_names = []
            for part_name in sorted(modality["state"].keys()):
                part_start = modality["state"][part_name]["start"]
                part_end = modality["state"][part_name]["end"]
                for i in range(part_end - part_start):
                    state_names.append(f"{part_name}_{i}")
            if state_names:
                features["observation.state"]["names"] = state_names

    if all_episodes and len(all_episodes[0]["actions"]) > 0:
        action_dim = all_episodes[0]["actions"].shape[-1]
        features["action"] = {
            "dtype": "float64",
            "shape": [int(action_dim)],
        }

        if modality.get("action"):
            action_names = []
            for part_name in sorted(modality["action"].keys()):
                part_start = modality["action"][part_name]["start"]
                part_end = modality["action"][part_name]["end"]
                for i in range(part_end - part_start):
                    action_names.append(f"{part_name}_{i}")
            if action_names:
                features["action"]["names"] = action_names

    features["timestamp"] = {"dtype": "float64", "shape": [1]}
    features["annotation.human.action.task_description"] = {
        "dtype": "int64",
        "shape": [1],
    }
    features["task_index"] = {"dtype": "int64", "shape": [1]}
    features["annotation.human.validity"] = {"dtype": "int64", "shape": [1]}
    features["episode_index"] = {"dtype": "int64", "shape": [1]}
    features["index"] = {"dtype": "int64", "shape": [1]}
    features["next.reward"] = {"dtype": "float64", "shape": [1]}
    features["next.done"] = {"dtype": "bool", "shape": [1]}

    total_videos = 0
    if all_episodes:
        for episode_data in all_episodes:
            if "pixels" in episode_data and episode_data["pixels"]:
                total_videos += len(episode_data["pixels"])

    robot_type = "Unknown"
    if all_env_metadata and "env_id" in all_env_metadata[0]:
        env_id = all_env_metadata[0]["env_id"]
        if "Reachy" in env_id:
            robot_type = "Reachy"
        elif "GR" in env_id or "GR1" in env_id:
            robot_type = "GR1ArmsOnly"
        else:
            robot_type = env_id.split("_")[0] if "_" in env_id else env_id

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": len(all_episodes),
        "total_frames": total_timesteps,
        "total_tasks": len(unique_tasks),
        "total_videos": total_videos,
        "total_chunks": 0,
        "chunks_size": 1000,
        "fps": 20.0,
        "splits": {
            "train": "0:100",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print("\nConversion complete!")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Tasks: {len(unique_tasks)}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert RecordingWrapper sessions to LeRobot format")
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help="Directory containing session subdirectories",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="joint",
        choices=["ik", "joint"],
        help="Action type for output dataset: 'ik' for OSC/IK commands, 'joint' for absolute joint positions",
    )

    args = parser.parse_args()

    convert_sessions_to_lerobot(
        args.input_dir,
        args.output_dir,
        action_type=args.action_type,
    )


if __name__ == "__main__":
    main()
