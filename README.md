# LiDAR Registration with Visual Foundation Models
[**arXiv**](https://arxiv.org/abs/2502.19374) | [**Website**](https://vfm-registration.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/YicKCR-iLlk)


This repository is the official implementation of the paper:

> **LiDAR Registration with Visual Foundation Models**
>
> [Niclas Vödisch](https://vniclas.github.io/), [Giovanni Cioffi](https://giovanni-cioffi.netlify.app/), [Marco Cannici](https://marcocannici.github.io/), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Davide Scaramuzza](https://rpg.ifi.uzh.ch/people_scaramuzza.html). <br>
>
> *Robotics: Science and Systems*, 2025

<p align="center">
  <img src="./assets/overview.png" alt="Overview of our approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{voedisch2025lidar,
  title={LiDAR Registration with Visual Foundation Models},
  author={Niclas Vödisch, Giovanni Cioffi, Marco Cannici, Wolfram Burgard, Davide Scaramuzza},
  journal={arXiv preprint arXiv:2502.19374},
  year={2025}
}
```


## 📔 Abstract

LiDAR registration is a fundamental task in robotic mapping and localization. A critical component of aligning two point clouds is identifying robust point correspondences using point descriptors. This step becomes particularly challenging in scenarios involving domain shifts, seasonal changes, and variations in point cloud structures. These factors substantially impact both handcrafted and learning-based approaches. In this paper, we address these problems by proposing to use DINOv2 features, obtained from surround-view images, as point descriptors. We demonstrate that coupling these descriptors with traditional registration algorithms, such as RANSAC or ICP, facilitates robust 6DoF alignment of LiDAR scans with 3D maps, even when the map was recorded more than a year before. Although conceptually straightforward, our method substantially outperforms more complex baseline techniques. In contrast to previous learning-based point descriptors, our method does not require domain-specific retraining and is agnostic to the point cloud structure, effectively handling both sparse LiDAR scans and dense 3D maps. We show that leveraging the additional camera data enables our method to outperform the best baseline by +24.8 and +17.3 registration recall on the NCLT and Oxford RobotCar datasets.


## 👩‍💻 Code

### 🏗 Setup

#### 🐋 Installation

Tested with `Docker version 27.2.1` and `Docker Compose version v2.29.2`.

- Download the model weights of the baselines (requires [gdown](https://pypi.org/project/gdown/) in python environment): `python src/vfm-reg/src/download_baseline_models.py`
- To build the image, run `docker compose build` in the root of this repository.
- Prepare using GUIs (e.g., rviz) in the container: `xhost +local:docker`
- Start container and mount data: `docker compose run -v PATH_TO_DATA:/data -it vfm-reg`
- Connect to a running container: `docker compose exec -it vfm-reg bash`
- More installations: `~/catkin_ws/src/vfm-reg/install.sh`
> Note: If you succeed in integrating these additional installation steps inside the docker file, please let us know.

#### 💻 Development - Githooks

We used multiple githooks during the development of this code. You can set up them using the following steps:

1. Outside the docker, create a venv or conda environment. Make sure to source that before committing.
2. Install requirements: `pip install -r dev_requirements.txt`
3. Install [pre-commit](https://pre-commit.com/) githook scripts: `pre-commit install`

Python formatter ([yapf](https://github.com/google/yapf), [iSort](https://github.com/PyCQA/isort)) settings can be set in [pyproject.toml](pyproject.toml).

This will automatically run the pre-commit hooks on every commit. You can skip this using the `--no-verify` flag, e.g., `commit -m "Update node" --no-verify`.
To run the githooks on all files independent of doing a commit, use `pre-commit run --all-files`.

### 💾 Download the data

#### NCLT Dataset

Website: https://robots.engin.umich.edu/nclt/

Please refer to the dataset's website for instructions how to download their data.
Then download the *images* and the *velodyne* data of the following scenes:
- 2012-01-08
- 2012-02-12
- 2012-03-17
- 2012-05-26
- 2012-10-28
- 2013-04-05

#### Oxford RobotCar Radar Dataset

> &#x26a0;&#xfe0f; **Note:** We do not include the model data in this code release. Please download them from the dataset's SDK: https://github.com/ori-mrg/robotcar-dataset-sdk/tree/master/models

Website: https://oxford-robotics-institute.github.io/radar-robotcar-dataset/

Please refer to the dataset's website for instructions how to download their data.
Then download the images of all cameras (Bumblebee XB3, Grasshopper 2 Left/Right/Rear) and the processed data for the left Velodyne HDL-32E LiDAR. To save storage, you can delete the images in the stereo folder except for the center camera. Please download the data of the following scenes:
- [2019-01-10 @ 11-46-21 GMT](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-10-11-46-21-radar-oxford-10k)
- [2019-01-15 @ 13-06-37 GMT](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-15-13-06-37-radar-oxford-10k)
- [2019-01-17 @ 14-03-00 GMT](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-17-14-03-00-radar-oxford-10k)
- [2019-01-18 @ 15-20-12 GMT](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-18-15-20-12-radar-oxford-10k)


### 🗃 Visualize the scenes <a name="scene_visualization"></a>

We provide a Open3D-based [tool](./src/vfm-reg/src/visualize_scenes.py) to visualize the registration scenes from the NCLT and Oxford RobotCar Radar datasets.
The scene files are stored in the `./scene/` folder of this repository.
For visualizing the scenes, run (replace `DATASET` with NCLT or RobotCar dataset):
- `python visualize_scenes.py /data/DATASET /scenes/DATASET` (visualize all scenes of dataset)
- `python visualize_scenes.py /data/DATASET /scenes/DATASET/scene_***.json` (visualize a specific scene)

### 🦖 Extract the point descriptors

This needs to be run only once. Creating the 3D scenes including the DINOv2-based point descriptors requires approximately 175GB for NCLT and 50GB for RobotCar. This process can take a few hours (~4 for each dataset). The `--output_folder` is an optional parameter, otherwise the data will be written to `/scenes/DATASET/processed_scenes`.

- `python prepare_scenes.py /data/DATASET /scenes/DATASET --output_folder /OUTPUT_DIR`

For testing, you can also provide a specific scene:

- `python prepare_scenes.py /data/DATASET /scenes/DATASET/scene_***.json --output_folder /OUTPUT_DIR`

### 🏃 Run the registration code

We use ROS only for visualization. Thus, you could easily remove it from the code base. Note that [simple visualization](#scene_visualization) of the scenes does not require ROS.

LiDAR-to-Scan Registration:
- Source catkin workspace: `source ~/catkin_ws/devel/setup.bash`
- Start a roscore in a separate terminal: `roscore`
- Run the registration: `rosrun vfm-reg registration_node.py /scenes/DATASET/processed_scenes`


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. Components of other works are released under their original license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

In our work and experiments, we have used components from many other works. We thank the authors for open-sourcing their code. In no specific order, we list source repositories:
- Oxford RobotCar Dataset: https://github.com/ori-mrg/robotcar-dataset-sdk
- NCLT Dataset: https://robots.engin.umich.edu/nclt/
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- FeatUp: https://github.com/mhamilton723/FeatUp
- FCGF: https://github.com/chrischoy/FCGF
- DIP: https://github.com/fabiopoiesi/dip
- GeDi: https://github.com/fabiopoiesi/gedi
- GCL: https://github.com/liuQuan98/GCL
- PointDSC: https://github.com/XuyangBai/PointDSC
- SpinNet: https://github.com/QingyongHu/SpinNet


This work was partially supported by a fellowship of the German Academic Exchange Service (DAAD). Niclas Vödisch acknowledges travel support from the European Union’s Horizon 2020 research and innovation program under ELISE grant agreement No. 951847 and from the ELSA Mobility Program within the project European Lighthouse On Safe And Secure AI (ELSA) under the grant agreement No. 101070617.
