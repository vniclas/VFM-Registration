# LiDAR Registration with Visual Foundation Models
[**arXiv**](https://arxiv.org/abs/2502.19374) | [**Website**](https://vfm-registration.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/YicKCR-iLlk)


This repository is the official implementation of the paper:

> **LiDAR Registration with Visual Foundation Models**
>
> [Niclas Vödisch](https://vniclas.github.io/), [Giovanni Cioffi](https://giovanni-cioffi.netlify.app/), [Marco Cannici](https://marcocannici.github.io/), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Davide Scaramuzza](https://rpg.ifi.uzh.ch/people_scaramuzza.html). <br>
>
> *arXiv preprint arXiv:2502.19374*, 2025

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

We will release the code upon acceptance of the paper.


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
