# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import skimage.morphology
import cv2
import numpy as np
import scipy
from collections import defaultdict
import torch
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel
import home_robot.utils.visualization as vu
import home_robot.utils.pose as pu
from PIL import Image, ImageDraw, ImageFont
from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
import matplotlib.pyplot as plt
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_category_id_to_coco_category,
    coco_categories_color_palette,
)
from home_robot.agent.NI_agent.NI_agent_module import NIAgentModule
from home_robot.agent.NI_agent.NI_matching import NIMatching
# For visualizing exploration issues
debug_frontier_map = False
from tqdm import tqdm
from home_robot.agent.NI_agent.NI_alfred_env import NIAlfredEnv

def construct_envs_single(config):
    files = json.load(open("data/alfred_data_small/splits/oct21.json"))[config.eval_split][config.from_idx:config.to_idx]
    env = NIAlfredEnv(config, files, 0)
    return env

class NIAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0, id2name=None):
        self.alfred_env = construct_envs_single(config)
        id2name = self.alfred_env.id_cat
        # self.max_steps = config.AGENT.max_steps
        self.max_steps = [1000, 1000, 1000, 1000, 1000]
        # self.max_steps = [500, 400, 300, 200, 200, 200, 200, 200, 200, 200, 200]
        # self.max_steps = [400, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_angle = config.AGENT.panorama_angle
            self.panorama_period = int(360 / self.panorama_angle) + 1
            self.panorama_start_steps = self.panorama_period * 3
        else:
            self.panorama_start_steps = 0
        # print("panorama_start_steps: ", self.panorama_start_steps)
        self.panorama_rotate_steps = int(360 / self.panorama_angle)

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)
        
        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=20,
                category_id_to_category_name=id2name,
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = NIMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self._module = NIAgentModule(
            config, matching=self.matching, instance_memory=self.instance_memory
        )

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes
        self.va_dist_limit = 1.5
        if (
            "planner_type" in config.AGENT.PLANNER
            and config.AGENT.PLANNER.planner_type == "old"
        ):
            print("Using old planner")
            from home_robot.navigation_planner.old_discrete_planner import (
                DiscretePlanner,
            )
        else:
            print("Using new planner")
            from home_robot.navigation_planner.discrete_planner import DiscretePlanner

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        # self.imagenav_visualizer = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )
        self.goal_pose = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering
        self.prev_task_type = None
        self.action_sequences = []
        self.action_success = []
        self.last_obs = None

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
        camera_angle: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
                the main inference function of the agent that lets it interact with
                vectorized environments.
        s
                This function assumes that the agent has been initialized.

                Args:
                    obs: current frame containing (RGB, depth, segmentation) of shape
                     (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
                    pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
                     of shape (num_environments, 3)
                    object_goal_category: semantic category of small object goals
                    camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

                Returns:
                    planner_inputs: list of num_environments planner inputs dicts containing
                        obstacle_map: (M, M) binary np.ndarray local obstacle map
                         prediction
                        sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                         and local map boundaries planning window (gx1, gx2, gy1, gy2)
                        goal_map: (M, M) binary np.ndarray denoting goal location
                    vis_inputs: list of num_environments visualization info dicts containing
                        explored_map: (M, M) binary np.ndarray local explored map
                         prediction
                        semantic_map: (M, M) np.ndarray containing local semantic map
                         predictions
        """
        # import pdb; pdb.set_trace()
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if obstacle_locations is not None:
            obstacle_locations = obstacle_locations.unsqueeze(1)
        if free_locations is not None:
            free_locations = free_locations.unsqueeze(1)
        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        (
            self.goal_map,
            self.found_goal,
            self.goal_pose,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            seq_obstacle_locations=obstacle_locations,
            seq_free_locations=free_locations,
            seq_camera_angle=camera_angle
        )
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]


        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if self.found_goal[0].item():
            goal_map = self._prep_goal_map_input()

        # found_goal = self.found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            if frontier_map is not None:
                self.semantic_map.update_frontier_map(
                    e, frontier_map[e][0].cpu().numpy()
                )
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            self.sub_task_timesteps[e][self.current_task_idx] += 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )


        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
                "goal_pose": self.goal_pose[e] if self.goal_pose is not None else None
            }
            for e in range(self.num_environments)
        ]
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.total_timesteps[e],
                }
                for e in range(self.num_environments)
            ]
            if self.record_instance_ids:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )
        else:
            vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [
            [0] * self.max_num_sub_task_episodes
        ] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()
        self._module.reset()

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self._module.reset_sub_episode()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset_for_env(e)
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0
        self.planner.reset()
        self._module.reset()
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

    def score_thresh(self, task_type):
        # If we have fully explored the environment, set the matching threshold to 0.0
        # to go to the highest scoring instance
        if self.fully_explored[0]:
            return 0.0

        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0
        

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        # import pdb; pdb.set_trace()
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]

        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            img_goal,
            camera_pose,
            keypoints,
            matches,
            confidence,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
            camera_angle,
        ) = self._preprocess_obs(obs, task_type)
        self.last_obs = obs
        info = {}
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            camera_pose=camera_pose,
            reject_visited_targets=self.reject_visited_targets,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
            camera_angle=camera_angle,
        )
        # import pdb; pdb.set_trace()

        # t2 = time.time()
        # print(f"Mapping and goal selection: {t2 - t1:.2f}")

        # 3 - Planning
        closest_goal_map = None
        dilated_obstacle_map = None
        short_term_goal = None
        could_not_find_path = False
        # if planner_inputs[0]["found_goal"]:
        #     self.episode_panorama_start_steps = 0
        planner_visual = None
        (
            action,
            closest_goal_map,
            short_term_goal,
            dilated_obstacle_map,
            could_not_find_path,
            planner_stop,
            planner_visual,
        ) = self.planner.plan(
            **planner_inputs[0],
            use_dilation_for_stg=self.use_dilation_for_stg,
            timestep=self.sub_task_timesteps[0][self.current_task_idx],
            debug=True
        )
        plt.figure(figsize=(20, 20))
        plt.subplot(331)
        plt.axis("off")
        plt.title("rgb")
        plt.imshow(obs.rgb / 255)
        plt.subplot(332)
        plt.axis("off")
        plt.title("depth")
        plt.imshow(obs.depth / 150 * 255)
        plt.subplot(333)
        plt.axis("off")
        plt.title("semantic")
        plt.imshow(obs.task_observations['semantic_frame'])
        plt.subplot(334)
        plt.axis("off")
        plt.title("explored")
        plt.imshow(self.semantic_map.get_explored_map(0))
        plt.subplot(335)
        plt.axis("off")
        plt.title("obstacle")
        plt.imshow(planner_inputs[0]["obstacle_map"])
        if planner_visual is not None:
            _stg = planner_visual["stg"]
            _pos = planner_visual["pos"]
            _goal = planner_visual["closest_goal_pt"]
            _navigable_goal_map = planner_visual["navigable_goal_map"]
            _goal_map = planner_visual["goal_map"]
            _fmm_dist = planner_visual["fmm_dist"]
            _traversible = planner_visual["traversible"]
            if _navigable_goal_map is not None:
                plt.subplot(336)
                plt.axis("off")
                plt.title("navigable_goal_map")
                plt.imshow(_navigable_goal_map)
                plt.plot(_stg[1], _stg[0], "bx")
                plt.plot(_pos[1], _pos[0], "rx")
            plt.subplot(337)
            plt.axis("off")
            plt.title("goal_map")
            plt.imshow(_goal_map)
            plt.plot(_stg[1], _stg[0], "bx")
            plt.plot(_pos[1], _pos[0], "rx")
            plt.subplot(338)
            plt.axis("off")
            plt.title("fmm_dist")
            plt.imshow(_fmm_dist)
            plt.plot(_stg[1], _stg[0], "bx")
            plt.plot(_pos[1], _pos[0], "rx")
            plt.plot(_goal[1], _goal[0], "gx")
            plt.subplot(339)
            plt.axis("off")
            plt.title("traversible")
            plt.imshow(_traversible)
            plt.plot(_stg[1], _stg[0], "bx")
            plt.plot(_pos[1], _pos[0], "rx")
            plt.plot(_goal[1], _goal[0], "gx")
        plt.show()
        plt.savefig(f"./datadump/planner_visual_{self.total_timesteps[0]}.jpg")
        plt.close()
            
            
        # t3 = time.time()
        # print(f"Planning: {t3 - t2:.2f}")

        if (
            self.sub_task_timesteps[0][self.current_task_idx]
            >= self.max_steps[self.current_task_idx]
        ):
            print("Reached max number of steps for subgoal, calling STOP")
            action = DiscreteNavigationAction.STOP

        if could_not_find_path and not planner_stop and action != DiscreteNavigationAction.STOP:
            # This doesn't help
            # print("Resetting explored area")
            # self.semantic_map.local_map[0, MC.EXPLORED_MAP] *= 0
            # self.semantic_map.global_map[0, MC.EXPLORED_MAP] *= 0

            # TODO: is this accurate?
            print("Can't find a path. Map fully explored.")
            self.fully_explored[0] = True
            self.force_match_against_memory = True

            # if self.reached_goal_candidate:
            #     # move to next sub-task
            #     # update semantic map
            #     # reset timesteps
            #     pass

        if self.visualize:
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            if task_type == "imagenav":
                collision = {"is_collision": False}
                info = {
                    **planner_inputs[0],
                    **vis_inputs[0],
                    "rgb_frame": obs.rgb,
                    "semantic_frame": obs.semantic,
                    "closest_goal_map": closest_goal_map,
                    "last_goal_image": obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                    "short_term_goal": short_term_goal,
                }
                if self.imagenav_visualizer is not None:
                    self.imagenav_visualizer.visualize(**info)
            else:
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                vis_inputs[0]["goal_name"] = goal_text_desc
                vis_inputs[0]["semantic_frame"] = obs.task_observations[
                    "semantic_frame"
                ]
                vis_inputs[0]["closest_goal_map"] = closest_goal_map
                vis_inputs[0]["third_person_image"] = obs.third_person_image
                vis_inputs[0]["short_term_goal"] = None
                vis_inputs[0]["instance_memory"] = self.instance_memory

                info = {
                    **planner_inputs[0],
                    **vis_inputs[0],
                    "short_term_goal": short_term_goal,
                }

        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.force_match_against_memory = False
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.reset_sub_episode()
        self.prev_task_type = task_type
        self.action_sequences.append(action)
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        # import matplotlib.pyplot as plt; plt.imsave("./datadump/rgb_preprocess.jpg", obs.rgb / 255)
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        current_task = obs.task_observations["tasks"][self.current_task_idx]
        current_goal_semantic_id = current_task["semantic_id"]

        semantic = obs.semantic
        instance_ids = None

        (
            matches,
            confidences,
            keypoints,
            local_instance_ids,
            all_matches,
            all_confidences,
            all_rgb_keypoints,
            instance_ids,
        ) = (None, None, None, None, [], [], [], [])

        if not self._module.instance_goal_found:
            if task_type == "imagenav":
                if self.goal_image is None:
                    img_goal = obs.task_observations["tasks"][self.current_task_idx][
                        "image"
                    ]
                    (
                        self.goal_image,
                        self.goal_image_keypoints,
                    ) = self.matching.get_goal_image_keypoints(img_goal)
                    # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.image_matching_function,
                    self.total_timesteps[0],
                    image_goal=self.goal_image,
                    goal_image_keypoints=self.goal_image_keypoints,
                    categories=[current_task["semantic_id"]],
                    use_full_image=False,
                )

            elif task_type == "languagenav":
                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.matching.match_language_to_image,
                    self.total_timesteps[0],
                    language_goal=current_task["description"],
                    categories=[current_task["semantic_id"]],
                    use_full_image=True,
                )
        
        # import pdb; pdb.set_trace()
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        # semantic shape: (num_environments, frame_height, frame_width, num_sem_categories)
        # semantic[a, b, c] is the one-hot encoding of the semantic category of pixel (b, c) in env a

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        # curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        curr_pose = obs.base_pose
        # print("current pose:", curr_pose)
        # print("last pose:", self.last_poses[0])
        # pose_delta = torch.from_numpy(obs.delta_pose).unsqueeze(0)
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        )
        pose_delta[-1] = torch.deg2rad(pose_delta[-1])
        pose_delta = pose_delta.unsqueeze(0)
        # print(f"Pose from env:\nprev_pose: {self.last_poses[0]}\ncurrent_pose: {curr_pose}\ndelta_pose: {pose_delta}\n")
        # if not (delta_pose - pose_delta).mean() <= 1e-3:
        #     breakpoint()
        # print("delta pose from env:", delta_pose)
        # print("delta pose calculated:", pose_delta)
        # print("delta of delta pose:", delta_pose - pose_delta)
        # import pdb; pdb.set_trace()
        self.last_poses[0] = curr_pose

        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0) if current_goal_semantic_id is not None else None

        # NOT USED AT ALL? ->
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        camera_angle = obs.camera_angle
        if camera_angle is not None:
            camera_angle = torch.tensor(np.asarray(camera_angle)).unsqueeze(0)
        # Match a goal against every instance in memory the moment we get it
        # or when the map just got fully explored
        if (
            task_type in ["languagenav", "imagenav"]
            and self.record_instance_ids
            and (
                self.sub_task_timesteps[0][self.current_task_idx] == 0
                or self.force_match_against_memory
            )
        ):
            if self.force_match_against_memory:
                print("Force a match against the memory")
            self.force_match_against_memory = False
            (all_rgb_keypoints, all_matches, all_confidences, instance_ids) = self._match_against_memory(
                task_type, current_task
            )

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            self.goal_image,
            camera_pose,
            keypoints,
            matches,
            confidences,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
            camera_angle,
        )

    def _match_against_memory(self, task_type: str, current_task: Dict, use_full_img=True):
        print("--------Matching against memory!--------")
        if task_type == "languagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.matching.match_language_to_image,
                self.total_timesteps[0],
                language_goal=current_task["description"],
                use_full_image=use_full_img,
                categories=[current_task["semantic_id"]] if current_task["semantic_id"] is not None else None,
            )
            stats = {
                i: {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "all": scores.flatten().tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_language_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        elif task_type == "imagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.image_matching_function,
                self.sub_task_timesteps[0][self.current_task_idx],
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
                use_full_image=use_full_img,
                categories=[current_task["semantic_id"]] if current_task["semantic_id"] is not None else None,
            )
            stats = {
                i: {
                    "mean": float(scores.sum(axis=1).mean()),
                    "median": float(np.median(scores.sum(axis=1))),
                    "max": float(scores.sum(axis=1).max()),
                    "min": float(scores.sum(axis=1).min()),
                    "all": scores.sum(axis=1).tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_image_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        return all_rgb_keypoints, all_matches, all_confidences, instance_ids

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map

        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue

            # cluster goal points
            try:
                c = DBSCAN(eps=4, min_samples=1)
                data = np.array(goal_map[e].nonzero()).T
                c.fit(data)

                # mask all points not in the largest cluster
                mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
                mode_mask = (c.labels_ != mode).nonzero()
                x = data[mode_mask]
                goal_map_ = np.copy(goal_map[e])
                goal_map_[x] = 0.0

                # adopt masked map if non-empty
                if goal_map_.sum() > 0:
                    goal_map[e] = goal_map_
            except Exception as e:
                print(e)
                return goal_map

        return goal_map
    
    
    def start_test(self, test_num=1000) -> None:
        for i in range(test_num):
            manual = True
            obs, info =  self.alfred_env.load_next_scene()
            last_obs = obs
            # semantic_vis_dump = f"{env.dump_path}/semantic_vis"
            # os.makedirs(semantic_vis_dump, exist_ok=True)
            self.reset()
            # t = 0
            stop = False
            self.panorama_start(obs)
            # all_rgb_keypoints, all_matches, all_confidences, instance_ids = self.open_vocab_alignment(object_name="knife")
            # all_rgb_keypoints, all_matches, all_confidences, instance_ids = self.open_vocab_alignment(object_name="vase")
            # print(instance_ids)
            while not self.alfred_env.is_over and not stop:
                # print("--------------------------Step---------------------------")
                action, info = self.act(obs)
                if manual:
                    act = input("Manual action (wad, s, t): ")
                    if act == "w":
                        action = DiscreteNavigationAction.MOVE_FORWARD
                    elif act == "s":
                        action = DiscreteNavigationAction.STOP
                    elif act == "a":
                        action = DiscreteNavigationAction.TURN_LEFT
                    elif act == "d":
                        action = DiscreteNavigationAction.TURN_RIGHT
                    elif act == "t":
                        manual = False
                obs, rew, stop, info, success = self.alfred_env.step_goat(action)
                last_obs = obs
                print("------------------------Step Done------------------------")
            
    def panorama_start(self, obs: Observations):
        """Act end-to-end."""
        # import pdb; pdb.set_trace()
        self.update_memory(obs)
        while self.total_timesteps[0] <= self.episode_panorama_start_steps:
            if self.total_timesteps[0] % self.panorama_period != 0:
                action = [DiscreteNavigationAction.TURN_RIGHT, self.panorama_angle]
            elif self.total_timesteps[0] != self.episode_panorama_start_steps:
                action = [DiscreteNavigationAction.LOOK_UP, 45]
            else:
                action = [DiscreteNavigationAction.LOOK_DOWN, 90]
            obs, rew, stop, info, success = self.alfred_env.step_goat(action)
            self.update_memory(obs)
            self.action_sequences.append(action)
            
    def env_init(self) -> None:
        self.last_obs, self.last_info = self.alfred_env.load_next_scene()
        
        
    def open_vocab_alignment(self, object_name: str=None, object_image: np.ndarray=None, semantic_id: int=None) -> int:
        '''
        Align the object name in the instruction to the detected objects in the environment
        Args:
            object_name: str
        Returns:
            cat_id: int
            obj_inst_id: int
        '''
        current_task = {}
        current_task['description'] = object_name
        current_task["semantic_id"] = semantic_id
        if object_image is not None:
            task_type = "imagenav"
            (self.goal_image, self.goal_image_keypoints,) = self.matching.get_goal_image_keypoints(object_image)
        elif object_name is not None:
            task_type = "languagenav"
        else:
            raise ValueError("Either object name or object image should be provided")
        (all_rgb_keypoints, all_matches, all_confidences, instance_ids) = self._match_against_memory(
                task_type, current_task, use_full_img=False
            )
        return all_rgb_keypoints, all_matches, all_confidences, instance_ids
            
    def nav_to(self, global_instance_id: int) -> bool:
        '''
        Navigate to the specified object in the environment, according to the object category and instance id
        Args:
            cat_id: int
            obj_inst_id: int
        Returns:
            success: bool
            error_info: str
        '''
        return True, ""
    
    def sub_goal_generation(self, instruction: str) -> List[List[str]]:
        '''
        Generate a sequence of subgoals for the given instruction
        Available actions: "PickupObject", "OpenObject", "PutObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "SliceObject"
        Args:
            instruction: str
        Returns:
            sub goal list: [[object name, "PickupObject"]]
        '''
        action_sequence = [[]]
        for action in action_sequence:
            object_name = action[0]
            object_action = self.alfred_env.inverse_action_mapping[action[1]]
            action_sequence.append([object_name, object_action])
        return action_sequence
    
    def get_obj_mask(self, global_instance_id: int, environment_id: int=0) -> np.ndarray:
        '''
        Extract the object mask for va_interaction of alfred
        Args:
            cat_id: int
            obj_inst_id: int
        Returns:
            object_mask: np.ndarray
        '''
        return self.instance_memory.instance_views[environment_id][global_instance_id]['instance_views'][-1].mask
    
    def update_memory(self, obs: Observations, info: Dict[str, Any]=None) -> None:
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            img_goal,
            camera_pose,
            keypoints,
            matches,
            confidence,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
            camera_angle,
        ) = self._preprocess_obs(obs, task_type)
        self.last_obs = obs
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            camera_pose=camera_pose,
            reject_visited_targets=self.reject_visited_targets,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
            camera_angle=camera_angle,
        )
    
    def look_at(self, cat_id: int, obj_inst_id: int) -> bool:
        '''
        Adjust the orientation of the robot and camera to look at the object
        '''
        execution_success = True
        object_distance = 0
        error_info = ""
        return True, object_distance, error_info
        
    def distance_adjustment(self, cat_id: int, obj_inst_id: int) -> bool:
        '''
        Adjust the distance between the robot and the object
        Returns:
            success: bool
            error_info: str
        '''
        execution_success, object_distance, error_info = self.look_at(cat_id, obj_inst_id)
        assert object_distance <= self.va_dist_limit
        return execution_success, object_distance, error_info
    
    
    def va_interaction_preparation(self, cat_id, obj_inst_id):
        look_success, distance, error = self.look_at(cat_id, obj_inst_id)
        if look_success:
            if distance > self.va_dist_limit:
                look_success, distance, error = self.distance_adjustment(cat_id, obj_inst_id)
                if not look_success:
                    print(f"Distance adjustment failed: {error}")
                    raise ValueError(f"Distance adjustment failed: {error}")
            return True
        else:
            return False
        
    
    def sub_goal_execution(self, sub_goal: List[str], goal_pointer: int) -> int:
        try:
            current_pointer = 0
            for i in range(goal_pointer, len(sub_goal)):
                print(f"Executing subgoal {i}: {sub_goal[i]}")
                current_action = sub_goal[i]
                object_name = current_action[0]
                object_action = current_action[1]
                cat_id, obj_inst_id = self.open_vocab_alignment(object_name)
                nav_success, error = self.nav_to(cat_id, obj_inst_id)
                if nav_success:
                    prep_status = self.va_interaction_preparation(cat_id, obj_inst_id)
                    obj_mask = self.get_obj_mask(cat_id, obj_inst_id)
                    obs, rew, done, info, success = self.alfred_env.step_goat(goat_action=[object_action, obj_mask])
                    self.update_memory(obs, info)
                    if success: 
                        current_pointer = i + 1
                    else:
                        raise ValueError(f"Subgoal execution failed: {error}")
                else:
                    raise ValueError(f"Navigation failed: {error}")
        except Exception as e:
            print(f"Subgoal execution failed: {e}")
        return current_pointer
                        
                    
                        
                    
            
            
    
    

