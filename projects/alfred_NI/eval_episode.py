import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint
import shutil
import numpy as np
from tqdm import tqdm
import cv2
from config_utils import get_config
# from habitat.core.env import Env

from home_robot.agent.NI_agent.NI_agent import NIAgent
import matplotlib as mpl
mpl.use("TkAgg")



if __name__ == "__main__":
    baseline_config_path = "projects/alfred_NI/configs/agent/alfred_eval.yaml"
    config = get_config("", baseline_config_path)

    # all_scenes = os.listdir(os.path.dirname(config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)) + "/content/")
    # all_scenes = sorted([x.split('.')[0] for x in all_scenes])

    # if args.scene_idx != -1:
    #     scene_start = args.scene_idx * 5
    #     config.habitat.dataset.content_scenes = all_scenes[scene_start:scene_start+5]

    # config.habitat.dataset.content_scenes = ["TEEsavR23oF"] # TODO: for debugging. REMOVE later.
    
    
    # import pdb;pdb.set_trace()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    if os.path.exists(config.DUMP_LOCATION):
        shutil.rmtree(config.DUMP_LOCATION)
    # env = construct_envs_single(config)
    # id2name = env.id_cat
    agent = NIAgent(config=config)
    # import pdb; pdb.set_trace()
    agent.start_test(1)