from alfred_env.env.thor_env_code import ThorEnvCode
import json
import numpy as np
import alfred_env.gen.constants as constants
import json
from typing import Optional, Tuple

# import habitat.config.default
import yaml
# from habitat_baselines.config.default import _BASELINES_CFG_DIR
# from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig, OmegaConf


def none_or_str(string):
    if string == '':
        return None
    else:
        return string
    
def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1
    
def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['repeat_idx']
    except:
        r_idx = 0
    language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])
    object_target = none_or_str(traj_data['pddl_params']['object_target'])
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])
    #toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])
    
    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced


class NIAlfredEnv(ThorEnvCode):
    def __init__(self, config, scene_names, rank=0):
        super().__init__(config, rank)
        self.config = config
        self.scene_names = scene_names
        self.scene_pointer = 0
        self.task_type = "languagenav"
        self.step_counter = 0
  
    def load_traj(self, scene_name):
        # import pdb; pdb.set_trace()
        json_dir = '/home/caozhe/workspace/FILM/alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        traj_data = json.load(open(json_dir))
        return traj_data, json_dir
    
    @property
    def is_over(self):
        return self.step_counter >= 1000
    
    def obs_proprecess(self, obs):
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]	
        instance_mask = self.event.instance_masks
        depth = self._preprocess_depth(depth)
        ds = constants.DETECTION_SCREEN_HEIGHT // self.config.ENVIRONMENT.frame_height
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds//2::ds, ds//2::ds]
        return obs
    
    def load_scene(self,):
        traj_data, data_path = self.load_traj(self.scene_names[self.scene_pointer])
        r_idx = self.scene_names[self.scene_pointer]['repeat_idx']
        self.traj_data = traj_data
        self.r_idx = r_idx
        instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        instruction = instruction.lower()
        task_type = get_arguments(traj_data)[1]
        obs, info = self.setup_scene(traj_data, task_type, r_idx, self.config)
        obs = self.obs_proprecess(obs)
        return obs, info

    def load_next_scene(self, ):
        obs, info = self.load_scene()
        return obs, info
    
    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0] * 1
        mask_err_below = depth <0.0
        depth[mask_err_below] = 100.0
        depth = depth * 100
        return depth
        
    
    def step_goat(self, goat_action):
        self.step_counter += 1
        action = self.goat_action_translate(goat_action)
        if action is not None:
            obs, rew, done, info, self.event, action = self.to_thor_api_exec(action=action, smooth_nav=False)
        obs = self.obs_proprecess(obs)
        return obs, info

        
def get_config(
    habitat_config_path: str,
    baseline_config_path: str,
    opts: Optional[list] = None,
    # configs_dir: str = _BASELINES_CFG_DIR,
) -> Tuple[DictConfig, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    # habitat_config = get_habitat_config(
    #     habitat_config_path, overrides=opts, configs_dir=configs_dir
    # )
    baseline_config = OmegaConf.load(baseline_config_path)
    config = DictConfig({
        # **habitat_config, 
        **baseline_config
        })

    # Ensure consistency between configurations inherited from the task
    # # and defaults and our code's configuration

    # sim_sensors = config.habitat.simulator.agents.main_agent.sim_sensors

    # rgb_sensor = sim_sensors.rgb_sensor
    # depth_sensor = sim_sensors.depth_sensor
    # semantic_sensor = sim_sensors.semantic_sensor
    frame_height = config.ENVIRONMENT.frame_height
    # assert rgb_sensor.height == depth_sensor.height
    # if semantic_sensor:
    #     assert rgb_sensor.height == semantic_sensor.height
    # assert rgb_sensor.height >= frame_height and rgb_sensor.height % frame_height == 0

    frame_width = config.ENVIRONMENT.frame_width
    # assert rgb_sensor.width == depth_sensor.width
    # if semantic_sensor:
    #     assert rgb_sensor.width == semantic_sensor.width
    # assert rgb_sensor.width >= frame_width and rgb_sensor.width % frame_width == 0

    camera_height = config.ENVIRONMENT.camera_height
    # assert camera_height == rgb_sensor.position[1]
    # assert camera_height == depth_sensor.position[1]
    # if semantic_sensor:
    #     assert camera_height == semantic_sensor.position[1]

    hfov = config.ENVIRONMENT.hfov
    # assert hfov == rgb_sensor.hfov
    # assert hfov == depth_sensor.hfov
    # if semantic_sensor:
    #     assert hfov == semantic_sensor.hfov

    # assert config.ENVIRONMENT.min_depth == depth_sensor.min_depth
    # assert config.ENVIRONMENT.max_depth == depth_sensor.max_depth
    # assert config.ENVIRONMENT.turn_angle == config.habitat.simulator.turn_angle

    return config



if __name__ == "__main__":
    import json
    files = json.load(open("/home/caozhe/workspace/FILM/alfred_data_small/splits/oct21.json"))["valid_unseen"][0:10]
    config_path = "projects/alfred_NI/configs/agent/alfred_eval.yaml"
    config = get_config("", config_path)
    env = NIAlfredEnv(config, files, 0)
    env.load_next_scene()