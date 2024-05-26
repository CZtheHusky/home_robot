from alfred_benchmark.alfred_env.thor_env_code import ThorEnvCode, get_new_pose
import json
import numpy as np
import alfred_benchmark.gen.constants as constants
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.perception.detection.utils import overlay_masks, filter_depth
from torchvision import transforms
from PIL import Image
import random
import os
import cv2




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
        self.res = transforms.Compose([transforms.ToPILImage(), transforms.Resize((config.frame_height, config.frame_width), interpolation = Image.NEAREST)])
        # self.vocab_raw = set(constants.OBJECTS_SINGULAR)
        objects_list = constants.OBJECTS_SINGULAR
        self.vocab_raw = set(objects_list)
        self.id_cat = {i: objects_list[i] for i in range(len(objects_list))}
        self.cat_id = {v: k for k, v in self.id_cat.items()}
        self.vocab_num = len(self.id_cat)
        self.ground_truth = True
        self.depth_threshold = [0.5, 1.5]
        self.instance_ignored = set()
        self.agent_pose = np.zeros(3)
        self.delta_pose = np.zeros(3)
        self.step_count = 0
        self.turn_angle = config.ENVIRONMENT.turn_angle
        self.frame_height = config.ENVIRONMENT.frame_height
        self.frame_width = config.ENVIRONMENT.frame_width
        self.hfov = config.ENVIRONMENT.hfov
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.max_steps = config.AGENT.max_steps
        self.action_mapping = {
            DiscreteNavigationAction.STOP: None,
            DiscreteNavigationAction.MOVE_FORWARD: "MoveAhead",
            DiscreteNavigationAction.TURN_LEFT: "RotateLeft",
            DiscreteNavigationAction.TURN_RIGHT: "RotateRight",
            DiscreteNavigationAction.LOOK_UP: "LookUp",
            DiscreteNavigationAction.LOOK_DOWN: "LookDown",
            DiscreteNavigationAction.PICK_OBJECT: "PickupObject",
            DiscreteNavigationAction.PLACE_OBJECT: "PutObject",
            DiscreteNavigationAction.OPEN_OBJECT: "OpenObject",
            DiscreteNavigationAction.CLOSE_OBJECT: "CloseObject",
            DiscreteNavigationAction.TOGGLE_OBJECT_ON: "ToggleObjectOn",
            DiscreteNavigationAction.TOGGLE_OBJECT_OFF: "ToggleObjectOff",
            DiscreteNavigationAction.SLICE_OBJECT: "SliceObject",
        }
        self.inverse_action_mapping = {
            "MoveAhead": DiscreteNavigationAction.MOVE_FORWARD,
            "RotateLeft": DiscreteNavigationAction.TURN_LEFT,
            "RotateRight": DiscreteNavigationAction.TURN_RIGHT,
            "LookUp": DiscreteNavigationAction.LOOK_UP,
            "LookDown": DiscreteNavigationAction.LOOK_DOWN,
            "PickupObject": DiscreteNavigationAction.PICK_OBJECT,
            "PutObject": DiscreteNavigationAction.PLACE_OBJECT,
            "OpenObject": DiscreteNavigationAction.OPEN_OBJECT,
            "CloseObject": DiscreteNavigationAction.CLOSE_OBJECT,
            "ToggleObjectOn": DiscreteNavigationAction.TOGGLE_OBJECT_ON,
            "ToggleObjectOff": DiscreteNavigationAction.TOGGLE_OBJECT_OFF,
            "SliceObject": DiscreteNavigationAction.SLICE_OBJECT,
        }
  
    def load_traj(self, scene_name):
        # import pdb; pdb.set_trace()
        json_dir = self.config.ALFRED_DATA_PATH + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        traj_data = json.load(open(json_dir))
        return traj_data, json_dir
    
    def segmentation_module(self, rgb):
        pass
    
    def depth_estimation(self, rgb):
        pass
    
    
    def filter_masks(self, masks: dict[str, np.ndarray]) -> dict:
        """Filter the instance masks with valid ones
        Args:
            masks (dict): key, value = object_id, mask
        Returns:
            new_masks (dict): key, value = object_id, mask    
        """
        new_masks = {}
        for inst_name, inst_mask in masks.items():
            # import pdb; pdb.set_trace()
            inst_name = inst_name.split('|')[0]
            inst_name = inst_name.split(".")[0].lower()
            if new_masks.get(inst_name, None) is None:
                new_masks[inst_name] = [inst_mask]
            else:
                new_masks[inst_name].append(inst_mask)
        valid = {}
        invalid = {}
        # os.makedirs(f"{self.dump_path}/masks", exist_ok=True)
        for inst_name, inst_mask_list in new_masks.items():
            if inst_name in self.vocab_raw:
                valid[inst_name] = inst_mask_list
            else:
                invalid[inst_name] = inst_mask_list
                self.instance_ignored.add(inst_name)
            for num_id, inst_mask in enumerate(inst_mask_list):
                mask_img = np.zeros(inst_mask.shape)
                mask_img[np.where(inst_mask)] = 255
                # image = Image.fromarray(mask_img.astype(np.uint8), 'L')
                # image.save(f"{self.dump_path}/masks/{self.step_counter}_{inst_name}_{num_id}.png")
        return valid, invalid
    
    @property
    def is_over(self):
        return self.step_counter >= 1000
    
    def obs_proprecess(
        self, 
        obs: np.ndarray,
        ) -> Observations:
        """Preprocess observations
        Args:
            obs (dict): raw observations, with rgbd, h, w
        Return:
            obs (dict): preprocessed observations
        
        """
        # os.makedirs(f"{self.dump_path}/rgbs", exist_ok=True)
        # os.makedirs(f"{self.dump_path}/segs", exist_ok=True)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(rgb.astype(np.uint8))
        # image.save(f"{self.dump_path}/rgbs/rgb{self.step_counter}.png")
        # image = Image.fromarray(self.event.instance_segmentation_frame)
        # image.save(f"{self.dump_path}/segs/seg{self.step_counter}.png")
        depth = obs[:, :, 3:4]	
        depth = self._preprocess_depth(depth)
        # ds = constants.DETECTION_SCREEN_HEIGHT // self.config.ENVIRONMENT.frame_height
        # if ds != 1:
        #     rgb = np.asarray(self.res(rgb.astype(np.uint8)))
        #     depth = depth[ds//2::ds, ds//2::ds]
        # import pdb; pdb.set_trace()
        obs_rt = Observations(rgb, depth)
        # test_goal = self.cat_id["coffeemachine"]
        test_goal = self.cat_id["sink"]
        obs_rt.task_observations = {"tasks": [{'type': self.task_type, 'semantic_id': test_goal, "description": self.id_cat[test_goal]}]}
        instance_mask = self.event.instance_masks   # dict: key, value = object_id, mask
        valid_masks, invalid_masks = self.filter_masks(instance_mask)
        try:
            class_ids = []
            masks = []
            for inst_name, inst_mask_list in valid_masks.items():
                # print(inst_name)
                class_id = self.cat_id[inst_name]
                for inst_mask in inst_mask_list:
                    class_ids.append(class_id)
                    masks.append(inst_mask)
            assert len(class_ids) == len(masks)
            masks = np.array(masks)
            class_ids = np.array(class_ids)
            if not self.ground_truth:
                pass
            semantic_map, instance_map = overlay_masks(masks, class_ids, depth.shape)
            # the semantic map assigns class ids to different region of the image, instances of the same class shares the same id
            # the instance map assigns a local unique id to each instance
            # the class_ids is the list of class ids of all instances, one can revoer the class id of a instance with class_ids[instance_id]
            obs_rt.semantic = semantic_map.astype(int)
            obs_rt.task_observations["instance_map"] = instance_map
            obs_rt.task_observations["instance_classes"] = class_ids
            # the semantic frame is the rendered image with semantic labels on it
            obs_rt.task_observations['semantic_frame'] = self.event.instance_segmentation_frame
            obs_rt.base_pose = self.agent_pose
            obs_rt.delta_pose = self.delta_pose
            # import pdb; pdb.set_trace()
            obs_rt.camera_angle = float(self.event.metadata['agent']['cameraHorizon'])
        except Exception as e:
            # print(e)
            raise e
        # import pdb; pdb.set_trace()
        return obs_rt
    
    def load_scene(self,):
        traj_data, data_path = self.load_traj(self.scene_names[self.scene_pointer])
        r_idx = self.scene_names[self.scene_pointer]['repeat_idx']
        self.traj_data = traj_data
        self.r_idx = r_idx
        instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        instruction = instruction.lower()
        self.dump_path = f"./debug/{self.scene_pointer}_{instruction.replace(' ', '_')}"
        # os.makedirs(self.dump_path,exist_ok=True)
        task_type = get_arguments(traj_data)[1]
        obs, info = self.setup_scene(traj_data, task_type, r_idx, self.config)
        obs = self.obs_proprecess(obs)
        return obs, info

    def load_next_scene(self, ):
        self.agent_pos = np.zeros(3)
        self.delta_pose = np.zeros(3)
        self.step_counter = 0
        obs, info = self.load_scene()
        self.scene_pointer += 1
        return obs, info
    
    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0] * 1
        # mask_err_below = depth <0.0
        # depth[mask_err_below] = 100.0
        # depth = depth * 100
        return depth
    
    def goat_action_translate(self, goat_action):
        if isinstance(goat_action, list):
            if isinstance(goat_action[1], int):
                return f"{self.action_mapping[goat_action[0]]}_{goat_action[1]}" 
            else:
                return self.action_mapping[goat_action[0]], goat_action[1]
        else:
            return self.action_mapping[goat_action]
        
    
    def step_goat(self, goat_action):
        self.step_counter += 1
        action = self.goat_action_translate(goat_action)
        if action is not None:
            if isinstance(action, tuple):
                obs, rew, done, info, success, self.event, target_instance_id, error_msg, action = self.va_interact(action=action[0], interaction_mask=action[1], smooth_nav=False)
            else:
                obs, rew, done, info, success, self.event, target_instance_id, error_msg, action = self.va_interact(action=action, smooth_nav=False)
            self.delta_pose = np.array(info['sensor_pose'])
            self.agent_pose = get_new_pose(self.agent_pose, self.delta_pose)
            self.gt_position = [v for v in self.event.metadata['agent']['position'].values()]
            self.gt_rotation = [v for v in self.event.metadata['agent']['rotation'].values()]
            self.gt_pose = np.array([self.gt_position[0], self.gt_position[2], self.gt_rotation[1]])
            # print(f"Pose_GT:\ngt_pose: {self.gt_pose}\nrecentered: {self.gt_pose - self.orig_pose}\npose_delta: {self.gt_pose - self.prev_pose}")
            self.prev_pose = self.gt_pose
            obs = self.obs_proprecess(obs)
            return obs, rew, done, info, success
        else:
            return None, None, True, None, None

        
    