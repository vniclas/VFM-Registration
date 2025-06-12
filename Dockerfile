FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

# Preseed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select Europe" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata

# General tools
RUN apt-get update && apt-get install -y tmux git htop python-is-python3 python3-pip wget

# Python libraries
RUN pip install numpy==1.20.3

# Install ROS
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654  # Update ROS key
RUN apt-get update && apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get update && apt-get install -y curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y ros-noetic-desktop
RUN apt-get update && apt-get install -y python3-catkin-tools python3-rospy python3-rosdep ros-noetic-rviz ros-noetic-imu-tools ros-noetic-pcl-ros ros-noetic-tf2-geometry-msgs ros-noetic-tf2-sensor-msgs ros-noetic-ros-numpy
RUN rosdep init
RUN rosdep update

# Create catkin workspace and copy codes
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws
COPY ./src /root/catkin_ws/src/vfm-reg

# Install pip requirements
RUN pip install networkx pyyaml==5.4 wheel
RUN python3 -m pip install --upgrade pip==24.2
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.20
RUN pip install colour_demosaicing pillow==7.0.0
RUN pip install gdown

# Install FeatUp
RUN pip install ftfy regex
RUN pip install git+https://github.com/mhamilton723/CLIP.git
RUN pip install kornia omegaconf pytorch-lightning==2.2.0 timm==0.4.12 torchmetrics
# Install in install.sh script

# Install faiss
RUN pip install faiss-cpu
RUN apt-get update && apt-get install -y libgflags-dev intel-mkl-full
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/facebookresearch/faiss.git
WORKDIR /root/catkin_ws/src/faiss
RUN cmake -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=axv512 -DFAISS_USE_LTO=ON -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES='/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so;/usr/lib/x86_64-linux-gnu/libmkl_sequential.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;-lpthread;-lm;-ldl' -B build .
RUN make -C build -j faiss
RUN make -C build install
# Address bug: https://bugs.launchpad.net/ubuntu/+source/intel-mkl/+bug/1947626
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libmkl_def.so:/usr/lib/x86_64-linux-gnu/libmkl_avx2.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so

# Install KISS-ICP
RUN apt-get update && apt-get install -y build-essential libeigen3-dev libtbb-dev pybind11-dev ninja-build clang-format-9
RUN ln -s /usr/bin/clang-format-9 /usr/bin/clang-format
WORKDIR /root/catkin_ws/src/vfm-reg/kiss-icp
RUN make editable

# Install TEASER++
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
WORKDIR /root/catkin_ws/src/TEASER-plusplus
RUN mkdir build
WORKDIR /root/catkin_ws/src/TEASER-plusplus/build
RUN cmake -DTEASERPP_PYTHON_VERSION=3.8 ..
RUN make teaserpp_python
RUN pip install ./python
RUN pip install --ignore-installed open3d==0.18.0

# Install MinkowskiEngine
# Downgrade to avoid error
ENV CUDA_HOME /usr/local/cuda
RUN pip install setuptools==65.0.2
# RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps  # Installed in install.sh script

# Install GeDi
RUN pip install torchgeometry==0.1.2 tensorboard protobuf==3.20
WORKDIR /root/catkin_ws/src/vfm-reg/vfm-reg/src/gedi/backbones
RUN pip install ./pointnet2_ops_lib/

# Install pip dependencies
RUN pip install scikit-learn pandas h5py easydict hdbscan

# Build catkin workspace
WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin config --profile default --cmake-args -DCMAKE_BUILD_TYPE=Release'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin config --profile debug -x _debug --cmake-args -DCMAKE_BUILD_TYPE=Debug'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin build -cs'

# Clean-up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# # Setup environment and entrypoint
RUN touch /ros_entrypoint.sh
RUN echo "#!/bin/bash" >> /ros_entrypoint.sh
RUN echo "set -e" >> /ros_entrypoint.sh
RUN echo "source \"/opt/ros/noetic/setup.bash\" --" >> /ros_entrypoint.sh
RUN echo "source \"/root/catkin_ws/devel/setup.bash\"" >> /ros_entrypoint.sh
RUN echo "exec \"\$@\"" >> /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

WORKDIR /
SHELL ["bash", "--command"]
ENV SHELL /usr/bin/bash
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
