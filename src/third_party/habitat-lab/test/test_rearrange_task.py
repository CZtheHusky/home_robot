#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import json
import os.path as osp
import sys
import time
from glob import glob

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

import habitat
import habitat.datasets.rearrange.run_episode_generator as rr_gen
import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.tasks.rearrange.rearrange_sim
import habitat.tasks.rearrange.rearrange_task
import habitat.utils.env_utils
import habitat_sim
from habitat.config.default import _HABITAT_CFG_DIR, get_config
from habitat.core.embodied_task import Episode
from habitat.core.environments import get_env_class
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.composite_task import CompositeTask
from habitat.utils.geometry_utils import is_point_in_triangle
from habitat_baselines.config.default import get_config as baselines_get_config

CFG_TEST = "benchmark/rearrange/pick.yaml"
GEN_TEST_CFG = (
    "habitat-lab/habitat/datasets/rearrange/configs/test_config.yaml"
)
EPISODES_LIMIT = 6


def check_json_serialization(dataset: RearrangeDatasetV0):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = RearrangeDatasetV0()
    decoded_dataset.from_json(json_str)
    decoded_dataset.config = dataset.config
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_rearrange_dataset():
    dataset_config = get_config(CFG_TEST).habitat.dataset
    if not RearrangeDatasetV0.check_config_paths_exist(dataset_config):
        pytest.skip(
            "Please download ReplicaCAD RearrangeDataset Dataset to data folder."
        )

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.type, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    check_json_serialization(dataset)


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob(
            "habitat-baselines/habitat_baselines/config/rearrange/**/*.yaml",
            recursive=True,
        ),
    ),
)
def test_rearrange_baseline_envs(test_cfg_path):
    """
    Test the Habitat Baseline environments
    """
    config = baselines_get_config(
        test_cfg_path,
        [
            "habitat.dataset.split=val",
            "habitat_baselines.eval.split=val",
        ],
    )
    for _, agent_config in config.habitat.simulator.agents.items():
        if (
            agent_config.articulated_agent_type == "KinematicHumanoid"
            and not osp.exists(agent_config.motion_data_path)
        ):
            pytest.skip(
                "This test should only be run if we have the motion data files."
            )
    with habitat.config.read_write(config):
        config.habitat.gym.obs_keys = None
        config.habitat.gym.desired_goal_keys = []

    env_class = get_env_class(config.habitat.env_task)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

    with env:
        for _ in range(10):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, done, _ = env.step(  # type:ignore[assignment]
                    action=action
                )


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("habitat-lab/habitat/config/benchmark/rearrange/*"),
    ),
)
def test_composite_tasks(test_cfg_path):
    """
    Test for the Habitat composite tasks.
    """
    if not osp.isfile(test_cfg_path):
        return

    config = get_config(
        test_cfg_path,
        [
            "habitat.simulator.concur_render=False",
            "habitat.dataset.split=val",
        ],
    )
    if "task_spec" not in config.habitat.task:
        return

    if (
        config.habitat.dataset.data_path
        == "data/ep_datasets/bench_scene.json.gz"
    ):
        pytest.skip(
            "This config is only useful for examples and does not have the generated dataset"
        )

    with habitat.Env(config=config) as env:
        if not isinstance(env.task, CompositeTask):
            return

        pddl_path = osp.join(
            _HABITAT_CFG_DIR,
            config.habitat.task.task_spec_base_path,
            config.habitat.task.task_spec + ".yaml",
        )
        with open(pddl_path, "r") as f:
            domain = yaml.safe_load(f)
        if "solution" not in domain:
            return
        n_stages = len(domain["solution"])

        for task_idx in range(n_stages):
            env.reset()
            env.task.jump_to_node(task_idx, env.current_episode)
            env.step(env.action_space.sample())
            env.reset()


# NOTE: set 'debug_visualization' = True to produce videos showing receptacles and final simulation state
@pytest.mark.parametrize("debug_visualization", [False])
@pytest.mark.parametrize("num_episodes", [2])
@pytest.mark.parametrize("config", [GEN_TEST_CFG])
def test_rearrange_episode_generator(
    debug_visualization, num_episodes, config
):
    cfg = rr_gen.get_config_defaults()
    override_config = OmegaConf.load(config)
    cfg = OmegaConf.merge(cfg, override_config)
    assert isinstance(cfg, DictConfig)
    dataset = RearrangeDatasetV0()
    with rr_gen.RearrangeEpisodeGenerator(
        cfg=cfg, debug_visualization=debug_visualization
    ) as ep_gen:
        start_time = time.time()
        dataset.episodes += ep_gen.generate_episodes(num_episodes)

    # test serialization of freshly generated dataset
    check_json_serialization(dataset)

    logger.info(
        f"successful_ep = {len(dataset.episodes)} generated in {time.time()-start_time} seconds."
    )


@pytest.mark.skipif(
    not osp.exists("data/test_assets/"),
    reason="This test requires habitat-sim test assets.",
)
def test_receptacle_parsing():
    # 1. Load the parameterized scene
    sim_settings = habitat_sim.utils.settings.default_sim_settings.copy()
    sim_settings[
        "scene"
    ] = "data/test_assets/scenes/simple_room.stage_config.json"
    sim_settings["sensor_height"] = 0
    sim_settings["enable_physics"] = True
    cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
    with habitat_sim.Simulator(cfg) as sim:
        # load test assets
        sim.metadata_mediator.object_template_manager.load_configs(
            "data/test_assets/objects/chair.object_config.json"
        )
        # TODO: add an AO w/ receptacles also

        # test quick receptacle listing:
        list_receptacles = hab_receptacle.get_all_scenedataset_receptacles(sim)
        print(f"list_receptacles = {list_receptacles}")
        # receptacles from stage configs:
        assert (
            "receptacle_aabb_simpleroom_test"
            in list_receptacles["stage"][
                "data/test_assets/scenes/simple_room.stage_config.json"
            ]
        )
        assert (
            "receptacle_mesh_simpleroom_test"
            in list_receptacles["stage"][
                "data/test_assets/scenes/simple_room.stage_config.json"
            ]
        )
        # receptacles from rigid object configs:
        assert (
            "receptacle_aabb_chair_test"
            in list_receptacles["rigid"][
                "data/test_assets/objects/chair.object_config.json"
            ]
        )
        assert (
            "receptacle_mesh_chair_test"
            in list_receptacles["rigid"][
                "data/test_assets/objects/chair.object_config.json"
            ]
        )
        # TODO: receptacles from articulated object configs:
        # assert "" in list_receptacles["articulated"]

        # add the chair to the scene
        chair_template_handle = (
            sim.metadata_mediator.object_template_manager.get_template_handles(
                "chair"
            )[0]
        )
        chair_obj = (
            sim.get_rigid_object_manager().add_object_by_template_handle(
                chair_template_handle
            )
        )

        def randomize_obj_state():
            chair_obj.translation = np.random.random(3)
            chair_obj.rotation = habitat_sim.utils.common.random_quaternion()
            # TODO: also randomize AO state here

        # parse the metadata into Receptacle objects
        test_receptacles = hab_receptacle.find_receptacles(sim)

        # test the Receptacle instances
        num_test_samples = 10
        for receptacle in test_receptacles:
            # check for contents and correct type parsing
            if receptacle.name == "receptacle_aabb_chair_test":
                assert type(receptacle) is hab_receptacle.AABBReceptacle
            elif receptacle.name == "receptacle_mesh_chair_test.0000":
                assert (
                    type(receptacle) is hab_receptacle.TriangleMeshReceptacle
                )
            elif receptacle.name == "receptacle_aabb_simpleroom_test":
                assert type(receptacle) is hab_receptacle.AABBReceptacle
            elif receptacle.name == "receptacle_mesh_simpleroom_test.0000":
                assert (
                    type(receptacle) is hab_receptacle.TriangleMeshReceptacle
                )
            else:
                # TODO: add AO receptacles
                raise AssertionError(
                    f"Unknown Receptacle '{receptacle.name}' detected. Update unit test golden values if this is expected."
                )

            for _six in range(num_test_samples):
                randomize_obj_state()
                # check that parenting and global transforms are as expected:
                parent_object = None
                expected_global_transform = mn.Matrix4.identity_init()
                global_transform = receptacle.get_global_transform(sim)
                if receptacle.parent_object_handle is not None:
                    parent_object = None
                    if receptacle.parent_link is not None:
                        # articulated object
                        assert receptacle.is_parent_object_articulated
                        parent_object = sim.get_articulated_object_manager().get_object_by_handle(
                            receptacle.parent_object_handle
                        )
                        expected_global_transform = (
                            parent_object.get_link_scene_node(
                                receptacle.parent_link
                            ).absolute_transformation()
                        )
                    else:
                        # rigid object
                        assert not receptacle.is_parent_object_articulated
                        parent_object = sim.get_rigid_object_manager().get_object_by_handle(
                            receptacle.parent_object_handle
                        )
                        # NOTE: we use absolute transformation from the 2nd visual node (scaling node) and root of all render assets to correctly account for any COM shifting, re-orienting, or scaling which has been applied.
                        expected_global_transform = (
                            parent_object.visual_scene_nodes[
                                1
                            ].absolute_transformation()
                        )
                    assert parent_object is not None
                    assert np.allclose(
                        global_transform, expected_global_transform, atol=1e-06
                    )
                else:
                    # this is a stage Receptacle (global transform)
                    if type(receptacle) is not hab_receptacle.AABBReceptacle:
                        assert np.allclose(
                            global_transform,
                            expected_global_transform,
                            atol=1e-06,
                        )
                    else:
                        # NOTE: global AABB Receptacles have special handling here which is not explicitly tested. See AABBReceptacle.get_global_transform()
                        expected_global_transform = global_transform

                for _six2 in range(num_test_samples):
                    sample_point = receptacle.sample_uniform_global(
                        sim, sample_region_scale=1.0
                    )
                    expected_local_sample_point = (
                        expected_global_transform.inverted().transform_point(
                            sample_point
                        )
                    )
                    if type(receptacle) is hab_receptacle.AABBReceptacle:
                        # check that the world->local sample point is contained in the local AABB
                        assert receptacle.bounds.contains(
                            expected_local_sample_point
                        )
                    elif (
                        type(receptacle)
                        is hab_receptacle.TriangleMeshReceptacle
                    ):
                        # check that the local point is within a mesh triangle
                        in_mesh = False
                        for f_ix in range(
                            int(len(receptacle.mesh_data.indices) / 3)
                        ):
                            verts = receptacle.get_face_verts(f_ix)
                            if is_point_in_triangle(
                                expected_local_sample_point,
                                verts[0],
                                verts[1],
                                verts[2],
                            ):
                                in_mesh = True
                                break
                        assert (
                            in_mesh
                        ), "The point must belong to a triangle of the local mesh to be valid."
