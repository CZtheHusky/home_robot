#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.pick_task import RearrangePickTaskV1


@registry.register_task(name="RearrangePlaceTask-v0")
class RearrangePlaceTaskV1(RearrangePickTaskV1):
    def __init__(self, *args, config, dataset=None, **kwargs):
        self.is_nav_to_obj = False
        super().__init__(
            *args,
            config=config,
            dataset=dataset,
            **kwargs,
        )

    def _get_targ_pos(self, sim):
        return sim.get_targets()[1]

    def _should_prevent_grip(self, action_args):
        # Never allow regrasping
        return (
            not self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] >= 0
        )

    def _get_spawn_goals(self, episode):
        return episode.candidate_goal_receps

    def reset(self, episode: Episode):
        sim = self._sim
        # Remove whatever the agent is currently holding.
        sim.grasp_mgr.desnap(force=True)

        super().reset(episode, fetch_observations=False)

        abs_obj_idx = sim.scene_obj_ids[self.abs_targ_idx]
        self._picked_object_idx = self.abs_targ_idx

        sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)

        self.was_prev_holding = self.targ_idx

        sim.internal_step(-1)
        self._sim.maybe_update_robot()
        return self._get_observations(episode)
