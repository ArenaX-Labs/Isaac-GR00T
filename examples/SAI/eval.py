"""
Evaluate the fine-tuned GR00T N1.5 on the SAI Kitchen environments.
"""

import argparse
from pathlib import Path
from typing import Dict
import json_numpy

json_numpy.patch()

import gymnasium as gym
import numpy as np
import sai_mujoco
import requests


class GR00TClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_action(self, obs):
        response = requests.post(f"http://{self.host}:{self.port}/act", json={"observation": obs})
        if response.status_code == 200:
            action = response.json()
            return action
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}


def _get_qpos(robot) -> Dict[str, np.ndarray]:
    sim = robot.sim
    q = {}

    base_joints = robot.parts["base"].joint_names
    q["base"] = np.array([sim.data.get_joint_qpos(j) for j in base_joints])
    q["left_arm"] = np.array([sim.data.get_joint_qpos(j) for j in robot.parts["left_arm"].joint_names])
    q["right_arm"] = np.array([sim.data.get_joint_qpos(j) for j in robot.parts["right_arm"].joint_names])
    q["left_hand"] = np.array([sim.data.get_joint_qpos(j) for j in robot.parts["left_arm_gripper"].joint_names])
    q["right_hand"] = np.array([sim.data.get_joint_qpos(j) for j in robot.parts["right_arm_gripper"].joint_names])
    q["torso"] = np.array([sim.data.get_joint_qpos(j) for j in robot.parts["torso"].joint_names])
    return q


def build_gr00t_obs(obs, robot, task_description: str) -> Dict[str, np.ndarray]:
    """Construct the observation dict expected by reachy2_mobile_base data_config."""
    left_img = obs["pixels"]["cam_robot_0:agentview_left_rgb"]
    center_img = obs["pixels"]["cam_robot_0:agentview_center_rgb"]
    right_img = obs["pixels"]["cam_robot_0:agentview_right_rgb"]
    q = _get_qpos(robot)

    gr00t_obs = {
        "video.cam_robot_0:agentview_center_rgb": center_img[None, ...],
        "video.cam_robot_0:agentview_left_rgb": left_img[None, ...],
        "video.cam_robot_0:agentview_right_rgb": right_img[None, ...],
        "state.base": q["base"][None, ...],
        "state.torso": q["torso"][None, ...],
        "state.left_arm": q["left_arm"][None, ...],
        "state.right_arm": q["right_arm"][None, ...],
        "state.left_hand": q["left_hand"][None, ...],
        "state.right_hand": q["right_hand"][None, ...],
        "annotation.human.action.task_description": [task_description],
    }
    return gr00t_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="Reachy2IkKitchenDrawerOpen-beta",
        help="Gym env id",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory to save video recordings. If None, no videos are saved.",
    )
    args = parser.parse_args()

    from gr00t.eval.wrappers.video_recording_wrapper import (
        VideoRecorder,
        VideoRecordingWrapper,
    )

    env = gym.make(
        args.env,
        render_mode="rgb_array",
        deterministic_reset=False,
        use_cam_obs=True,
        env_obs_cam_names=[],
        robot_obs_cam_names=[
            "robot_0:agentview_center",
            "robot_0:agentview_left",
            "robot_0:agentview_right",
        ],
        offscreen_render_mode="rgb_array",
    )

    if args.video_dir is not None:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_recorder = VideoRecorder.create_h264(
            fps=30,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=18,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            mode="rgb_array",
            video_dir=video_dir,
            steps_per_render=1,
        )
        print(f"Video recording enabled. Videos will be saved to: {video_dir}")

    client = GR00TClient(host=args.host, port=args.port)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < args.max_steps:
            task_desc = info.get("task_desc")
            gr00t_obs = build_gr00t_obs(obs, env.unwrapped.robots[0], task_desc)
            action_chunk = client.get_action(gr00t_obs)

            env_action = np.concatenate(
                [
                    np.zeros(4),  # base and torso actions are not used
                    action_chunk["action.left_arm"][0],
                    action_chunk["action.right_arm"][0],
                    action_chunk["action.left_hand"][0],
                    action_chunk["action.right_hand"][0],
                ]
            ).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            step += 1

        success = info.get("success", False)
        print(f"Episode {ep + 1} finished, success={success}, steps={step}")


if __name__ == "__main__":
    main()
