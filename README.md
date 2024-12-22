# pytest

# install cuda
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-cudnn


# create venv
cd myproj
python3 -m venv venv 

# activate venv 
source venv/bin/activate



# install tensorflow
# e ci nette una vita
python3 -m 
pip install tensorflow[and-cuda] --force-reinstall --no-cache-dir
pip install ipykernel --force-reinstall --no-cache-dir
pip install seaborn --force-reinstall --no-cache-dir
pip install matplotlib --force-reinstall --no-cache-dir
pip install scikit-learn --force-reinstall --no-cache-dir

# Check if TensorFlow is using GPU
import tensorflow as tf
tf.config.list_physical_devices('GPU')


# NUMA node
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done




Ref 
# keras
Installing NVIDIA Drivers and CUDA on Linux Mint 21.3
https://iceburn.medium.com/installing-nvidia-drivers-and-cuda-on-linux-mint-21-3-16acdc0b0083

# Install TensorFlow with pip
https://www.tensorflow.org/install/pip




physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")
