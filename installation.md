```
conda env create -n alfred39 python==3.9.13 -y
pip install -r reqs.txt
pip install -r overwrite_reqs.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
git submodule update --init --recursive
pip install -e ./src/home_robot
pip install -e ./src/home_robot_sim
# headless
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
# with display
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
pip install -e ./src/third_party/habitat-lab/habitat-lab
pip install -e ./src/third_party/habitat-lab/habitat-baselines
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```