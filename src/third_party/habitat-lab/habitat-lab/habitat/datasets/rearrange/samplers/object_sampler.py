#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import is_accessible
from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    Receptacle,
    ReceptacleTracker,
    TriangleMeshReceptacle,
    find_receptacles,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.tasks.rearrange.utils import get_aabb


class ObjectSampler:
    """
    Sample an object from a set and try to place it in the scene on a Receptacles from some Receptacle set.
    """

    def __init__(
        self,
        object_set: List[str],
        allowed_recep_set_names: List[str],
        sampler_range_type: str,
        num_objects: Optional[Tuple[int, int]] = None,
        orientation_sample: Optional[str] = None,
        sample_region_ratio: Optional[Dict[str, float]] = None,
        nav_to_min_distance: float = -1.0,
        object_set_sample_probs: Optional[Dict[str, float]] = None,
        recep_set_sample_probs: Optional[Dict[str, float]] = None,
        translation_up_offset=0.08,
    ) -> None:
        """
        :param object_set: The set objects from which placements will be sampled.
        :param allowed_recep_set_names:
        :param num_objects: The [minimum, maximum] number of objects for this sampler. Actual target value for the sampler will be uniform random number in this range.
        :param orientation_sample: Optionally choose to sample object orientation as well as position. Options are: None, "up" (1D), "all" (rand quat).
        :param sample_region_ratio: Defines a XZ scaling of the sample region around its center. Default no scaling. Enables shrinking aabb receptacles away from edges.
        :param nav_to_min_distance: -1.0 means there will be no accessibility constraint. Positive values indicate minimum distance from sampled object to a navigable point.
        :param recep_set_sample_probs: Optionally provide a non-uniform weighting for receptacle sampling.
        :param translation_up_offset: Optionally offset sample points to improve likelyhood of successful placement on inflated collision shapes.
        """
        self.object_set = object_set
        self._allowed_recep_set_names = allowed_recep_set_names
        self._object_set_sample_probs = object_set_sample_probs
        self._recep_set_sample_probs = recep_set_sample_probs
        self._sampler_range_type = sampler_range_type
        self._translation_up_offset = translation_up_offset

        self.receptacle_instances: Optional[
            List[Receptacle]
        ] = None  # all receptacles in the scene
        self.receptacle_candidates: Optional[
            List[Receptacle]
        ] = None  # the specific receptacle instances relevant to this sampler
        self.max_sample_attempts = 100  # number of distinct object|receptacle pairings to try before giving up
        self.max_placement_attempts = 50  # number of times to attempt a single object|receptacle placement pairing
        # type of sampler range: dynamic, fixed.
        # - dynamic sets number of objects to sample based on total receptacle area
        # - fixed sets number of objects to sample to the given range
        if self._sampler_range_type == "dynamic":
            self._num_objects = None
        elif self._sampler_range_type == "fixed":
            self.num_objects = (
                num_objects  # tuple of [min,max] objects to sample
            )
            assert self.num_objects[1] >= self.num_objects[0]
            self.set_num_samples()
        else:
            raise ValueError(
                f"Sampler range type {self._sampler_range_type} not recognized."
            )
        self.orientation_sample = (
            orientation_sample  # None, "up" (1D), "all" (rand quat)
        )
        if sample_region_ratio is None:
            sample_region_ratio = defaultdict(lambda: 1.0)
        self.sample_region_ratio = sample_region_ratio
        self.nav_to_min_distance = nav_to_min_distance
        # More possible parameters of note:
        # - surface vs volume
        # - apply physics stabilization: none, dynamic, projection

    def reset(self) -> None:
        """
        Reset any per-scene variables.
        """
        # receptacle instances should be scraped for every new scene
        self.receptacle_instances = None
        self.receptacle_candidates = None
        # number of objects in the range should be reset each time
        if self._sampler_range_type == "dynamic":
            self._num_objects = None
        else:
            self.set_num_samples()

    def set_receptacle_instances(self, receptacles: List[Receptacle]) -> None:
        """
        Set the receptacle instances to sample from.
        """
        self.receptacle_instances = receptacles
        if self._sampler_range_type == "dynamic":
            total_receptacle_area = sum(rec.total_area for rec in receptacles)
            self.num_objects = (
                int(np.floor(total_receptacle_area * 1.5)),
                int(np.floor(total_receptacle_area * 2)),
            )
            self.set_num_samples()

    def sample_receptacle(
        self,
        sim: habitat_sim.Simulator,
        recep_tracker: ReceptacleTracker,
        cull_tilted_receptacles: bool = True,
        tilt_tolerance: float = 0.9,
    ) -> Receptacle:
        """
        Sample a receptacle from the receptacle_set and return relevant information.

        :param sim: The active Simulator instance.
        :param recep_tracker: The pre-initialized ReceptacleTracker object defining available ReceptacleSets.
        :param cull_tilted_receptacles: Whether or not to remove tilted Receptacles from the candidate set.
        :param tilt_tolerance: If cull_tilted_receptacles is True, receptacles are culled for objects with local "down" (-Y), not aligned with gravity (unit dot product compared to tilt_tolerance).

        :return: The sampled Receptacle. AssertionError if no valid Receptacle candidates are found.
        """
        if self.receptacle_instances is None:
            self.receptacle_instances = find_receptacles(sim)

        match_recep_sets = [
            recep_tracker.recep_sets[k] for k in self._allowed_recep_set_names
        ]

        if self._recep_set_sample_probs is not None:
            sample_weights = [
                self._recep_set_sample_probs[k]
                for k in self._allowed_recep_set_names
            ]
            match_recep_sets = random.choices(
                match_recep_sets, weights=sample_weights
            )

        if match_recep_sets[0].is_on_top_of_sampler:
            rs = match_recep_sets[0]
            return OnTopOfReceptacle(
                rs.name,
                rs.included_receptacle_substrings,
            )

        if self.receptacle_candidates is None:
            self.receptacle_candidates = []
            for receptacle in self.receptacle_instances:
                found_match = False
                for receptacle_set in match_recep_sets:
                    culled = False
                    # first try to cull by exclusion
                    for ex_object_substr in (
                        receptacle_set.excluded_object_substrings
                        and receptacle.parent_object_handle
                    ):
                        if ex_object_substr in receptacle.parent_object_handle:
                            culled = True
                            break
                    for (
                        ex_receptacle_substr
                    ) in receptacle_set.excluded_receptacle_substrings:
                        if ex_receptacle_substr in receptacle.unique_name:
                            culled = True
                            break
                    if culled:
                        break

                    # if the receptacle is stage/global (no object handle) then always a match
                    if receptacle.parent_object_handle is None:
                        # check the inclusion name constraints
                        for (
                            name_constraint
                        ) in receptacle_set.included_receptacle_substrings:
                            if name_constraint in receptacle.unique_name:
                                found_match = True
                                break
                        break

                    # then search for inclusion
                    for (
                        object_substr
                    ) in receptacle_set.included_object_substrings:
                        if object_substr in receptacle.parent_object_handle:
                            # object substring is valid, try receptacle name constraint
                            for (
                                name_constraint
                            ) in receptacle_set.included_receptacle_substrings:
                                if name_constraint in receptacle.unique_name:
                                    # found a valid substring match for this receptacle, stop the search
                                    found_match = True
                                    break
                        if found_match:
                            # break object substr search
                            break
                    if found_match:
                        # break receptacle set search
                        break

                if found_match:
                    # substring match was found, check orientation constraint
                    if cull_tilted_receptacles:
                        obj_down = (
                            receptacle.get_global_transform(sim)
                            .transform_vector(-receptacle.up)
                            .normalized()
                        )
                        gravity_alignment = mn.math.dot(
                            obj_down, sim.get_gravity().normalized()
                        )
                        if gravity_alignment < tilt_tolerance:
                            culled = True
                            logger.info(
                                f"Culled by tilt: '{receptacle.unique_name}', {gravity_alignment}"
                            )
                    if not culled:
                        # found a valid receptacle
                        self.receptacle_candidates.append(receptacle)

        assert (
            len(self.receptacle_candidates) > 0
        ), f"No receptacle instances found matching this sampler's requirements. Likely a sampler config constraint is not feasible for all scenes in the dataset. Cull this scene from your dataset? Scene='{sim.config.sim_cfg.scene_id}'. "
        target_receptacle = self.receptacle_candidates[
            random.randrange(0, len(self.receptacle_candidates))
        ]
        return target_receptacle

    def sample_object(self) -> str:
        """
        Sample an object handle from the object_set and return it.
        """
        if self._object_set_sample_probs is not None:
            sample_weights = [
                self._object_set_sample_probs[k] for k in self.object_set
            ]
            return random.choices(self.object_set, weights=sample_weights)[0]
        else:
            return self.object_set[random.randrange(0, len(self.object_set))]

    def sample_placement(
        self,
        sim: habitat_sim.Simulator,
        object_handle: str,
        receptacle: Receptacle,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Attempt to sample a valid placement of the object in/on a receptacle given an object handle and receptacle information.

        :param sim: The active Simulator instance.
        :param object_handle: The handle of the object template for instantiation and attempted placement.
        :param receptacle: The Receptacle instance on which to sample a placement position.
        :param snap_down: Whether or not to use the snap_down utility to place the object.
        :param vdb: Optionally provide a debug visualizer (vdb)

        :return: The newly instanced rigid object or None if placement sampling failed.
        """
        num_placement_tries = 0
        new_object = None
        navmesh_vertices = np.stack(
            sim.pathfinder.build_navmesh_vertices(), axis=0
        )
        # Note: we cache the largest island to reject samples which are primarily accessible from disconnected navmesh regions. This assumption limits sampling to the largest navigable component of any scene.
        self.largest_island_size = max(
            [sim.pathfinder.island_radius(p) for p in navmesh_vertices]
        )

        while num_placement_tries < self.max_placement_attempts:
            num_placement_tries += 1

            # sample the object location
            target_object_position = (
                receptacle.sample_uniform_global(
                    sim, self.sample_region_ratio[receptacle.name]
                )
                + self._translation_up_offset * receptacle.up
            )

            # instance the new potential object from the handle
            if new_object == None:
                assert sim.get_object_template_manager().get_library_has_handle(
                    object_handle
                ), f"Found no object in the SceneDataset with handle '{object_handle}'."
                new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
                    object_handle
                )

                # fail early if it's impossible to place the object
                cumulative_bb = new_object.root_scene_node.cumulative_bb
                new_object_base_area = (
                    cumulative_bb.size_x() * cumulative_bb.size_z()
                )
                if new_object_base_area > receptacle.total_area:
                    logger.info(
                        f"Failed to sample placement. {object_handle} was too large to place on {receptacle.name}"
                    )
                    return None

            # try to place the object
            new_object.translation = target_object_position
            if self.orientation_sample is not None:
                if self.orientation_sample == "up":
                    # rotate the object around the gravity direction
                    rot = random.uniform(0, math.pi * 2.0)
                    new_object.rotation = mn.Quaternion.rotation(
                        mn.Rad(rot), mn.Vector3.y_axis()
                    )
                elif self.orientation_sample == "all":
                    # set the object's orientation to a random quaternion
                    new_object.rotation = (
                        habitat_sim.utils.common.random_quaternion()
                    )
            if isinstance(receptacle, TriangleMeshReceptacle):
                new_object.translation = new_object.translation + mn.Vector3(
                    0, 0.05, 0
                )

            if isinstance(receptacle, OnTopOfReceptacle):
                snap_down = False
            if snap_down:
                support_object_ids = [-1]
                # add support object ids for non-stage receptacles
                if receptacle.is_parent_object_articulated:
                    ao_instance = sim.get_articulated_object_manager().get_object_by_handle(
                        receptacle.parent_object_handle
                    )
                    for (
                        object_id,
                        link_ix,
                    ) in ao_instance.link_object_ids.items():
                        if receptacle.parent_link == link_ix:
                            support_object_ids = [
                                object_id,
                                ao_instance.object_id,
                            ]
                            break
                elif receptacle.parent_object_handle is not None:
                    support_object_ids = [
                        sim.get_rigid_object_manager()
                        .get_object_by_handle(receptacle.parent_object_handle)
                        .object_id
                    ]
                snap_success = sutils.snap_down(
                    sim,
                    new_object,
                    support_object_ids,
                    vdb=vdb,
                )
                if snap_success:
                    logger.info(
                        f"Successfully sampled (snapped) object placement in {num_placement_tries} tries."
                    )
                    if not is_accessible(
                        sim, new_object.translation, self.nav_to_min_distance
                    ):
                        logger.warning(
                            f"Failed to navigate to {object_handle} on {receptacle.name} in {num_placement_tries} tries."
                        )
                        continue
                    object_aabb = get_aabb(
                        new_object.object_id, sim, transformed=True
                    )
                    object_corners = [
                        object_aabb.back_bottom_left,
                        object_aabb.back_bottom_right,
                        object_aabb.front_bottom_left,
                        object_aabb.front_bottom_right,
                    ]
                    if not all(
                        receptacle.check_if_point_on_surface(
                            sim, corner, threshold=0.1
                        )
                        for corner in object_corners
                    ):
                        logger.warning(
                            f"Failed to place {object_handle} within bounds of {receptacle.name} in {num_placement_tries} tries."
                        )
                        continue
                    return new_object

            elif not new_object.contact_test():
                logger.info(
                    f"Successfully sampled object placement in {num_placement_tries} tries."
                )
                if not is_accessible(
                    sim, new_object.translation, self.nav_to_min_distance
                ):
                    continue
                return new_object

        # if num_placement_tries > self.max_placement_attempts:
        sim.get_rigid_object_manager().remove_object_by_handle(
            new_object.handle
        )
        logger.warning(
            f"Failed to sample {object_handle} placement on {receptacle.unique_name} in {self.max_placement_attempts} tries."
        )

        return None

    def single_sample(
        self,
        sim: habitat_sim.Simulator,
        recep_tracker: ReceptacleTracker,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
        fixed_target_receptacle=None,
        fixed_obj_handle: Optional[str] = None,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Sample a single object placement by first sampling a Receptacle candidate, then an object, then attempting to place that object on the Receptacle.

        :param sim: The active Simulator instance.
        :param recep_tracker: The pre-initialized ReceptacleTracker instace containg active ReceptacleSets.
        :param snap_down: Whether or not to use the snap_down utility to place the objects.
        :param vdb: Optionally provide a debug visualizer (vdb)
        :param fixed_target_receptacle: Optionally provide a pre-selected Receptacle instead of sampling. For example, when a target object's receptacle is selected in advance.
        :param fixed_obj_handle: Optionally provide a pre-selected object instead of sampling. For example, when sampling the goal position for a known target object.

        :return: The newly instanced rigid object or None if sampling failed.
        """
        # draw a new pairing
        if fixed_obj_handle is None:
            object_handle = self.sample_object()
        else:
            object_handle = fixed_obj_handle
        if fixed_target_receptacle is not None:
            target_receptacle = fixed_target_receptacle
        else:
            target_receptacle = self.sample_receptacle(sim, recep_tracker)
        logger.info(
            f"Sampling '{object_handle}' from '{target_receptacle.unique_name}'"
        )

        new_object = self.sample_placement(
            sim, object_handle, target_receptacle, snap_down, vdb
        )

        return new_object, target_receptacle

    def set_num_samples(self) -> None:
        """
        Choose a target number of objects to sample from the configured range.
        """
        self.target_objects_number = (
            random.randrange(self.num_objects[0], self.num_objects[1])
            if self.num_objects[1] > self.num_objects[0]
            else self.num_objects[0]
        )

    def sample(
        self,
        sim: habitat_sim.Simulator,
        recep_tracker: ReceptacleTracker,
        target_receptacles: List[Receptacle],
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> List[Tuple[habitat_sim.physics.ManagedRigidObject, Receptacle]]:
        """
        Defaults to uniform sample: object -> receptacle -> volume w/ rejection -> repeat.

        :param sim: The active Simulator instance.
        :param recep_tracker: The pre-initialized ReceptacleTracker instace containg active ReceptacleSets.
        :param target_receptacles: A list of pre-selected Receptacles for target object placement. These will be sampled first.
        :param snap_down: Whether or not to use the snap_down utility to place the objects.
        :param vdb: Optionally provide a debug visualizer (vdb)

        :return: The list of new (object,receptacle) pairs placed by the sampler.
        """
        num_pairing_tries = 0
        new_objects: List[
            Tuple[habitat_sim.physics.ManagedRigidObject, Receptacle]
        ] = []

        logger.info(
            f"    Trying to sample {self.target_objects_number} from range {self.num_objects}"
        )

        sampling_start_time = time.time()
        pairing_start_time = sampling_start_time
        while (
            len(new_objects) < self.target_objects_number
            and num_pairing_tries < self.max_sample_attempts
        ):
            num_pairing_tries += 1
            if len(new_objects) < len(target_receptacles):
                # sample objects explicitly from pre-designated target receptacles first
                new_object, receptacle = self.single_sample(
                    sim,
                    recep_tracker,
                    snap_down,
                    vdb,
                    target_receptacles[len(new_objects)],
                )
                # This receptacle has already been counted in the receptacle
                # tracking so don't double count.
            else:
                new_object, receptacle = self.single_sample(
                    sim, recep_tracker, snap_down, vdb
                )
                if (
                    new_object is not None
                    and recep_tracker.allocate_one_placement(receptacle)
                ):
                    # used up receptacle, need to recompute the sampler's receptacle_candidates
                    self.receptacle_candidates = None

            if new_object is not None:
                # when an object placement is successful, reset the try counter.
                logger.info(
                    f"    found obj|receptacle pairing ({len(new_objects)}/{self.target_objects_number}) in {num_pairing_tries} attempts ({time.time()-pairing_start_time}sec)."
                )
                num_pairing_tries = 0
                pairing_start_time = time.time()
                new_objects.append((new_object, receptacle))

        logger.info(
            f"    Sampling process completed in ({time.time()-sampling_start_time}sec)."
        )

        if len(new_objects) >= self.num_objects[0]:
            return new_objects

        # we didn't find the minimum number of placements, so remove all new objects and return
        logger.info(
            f"Failed to sample the minimum number of placements in {self.max_sample_attempts} tries."
        )
        logger.info(
            f"    Only able to sample {len(new_objects)} out of {self.num_objects}..."
        )
        # cleanup
        for new_object, _ in new_objects:
            sim.get_rigid_object_manager().remove_object_by_handle(
                new_object.handle
            )
        return []
