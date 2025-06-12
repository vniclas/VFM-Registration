pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

pip install git+https://github.com/mhamilton723/FeatUp
cp ~/catkin_ws/src/vfm-reg/featup_data/bpe_simple_vocab_16e6.txt.gz /usr/local/lib/python3.8/dist-packages/featup/featurizers/maskclip/

cd /root/catkin_ws/src/vfm-reg/kiss-icp && make editable
