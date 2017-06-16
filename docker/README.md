# Training and evaluating Tensorflow models using docker images

## Prerequisites (Ubuntu)

### docker

If you don't plan to use your GPU, you can install the Ubuntu docker packages.

However, if you need GPU support, you need to install docker-ce from the
[docker official website](https://docs.docker.com/engine/installation/linux/ubuntu/)

### nvidia driver with cuda support

Depending on your setup, your mileage may vary, but this worked for me:

~~~~
$ sudo apt-get install ubuntu-drivers-common
$ ubuntu-drivers devices
…
model    : GK107 [GeForce GTX 650]
…
driver   : nvidia-375 - distro non-free recommended
…
$ sudo apt-get install nvidia-375
$ sudo apt-get install nvidia-modprobe
~~~~

### nvidia-docker

The nvidia-docker wrapper is required to get access to the GPU from inside a
container.

Please follow [the official instructions](https://github.com/NVIDIA/nvidia-docker#quick-start).

## Available images

### Base images

Official Google [images](https://hub.docker.com/r/tensorflow/tensorflow/).

### Custom build

Since Google only provides generic images for x86 processors, you may not be
able to benefit from specific processor optimizations (like AVX instructions).

Your only option is therefore to fetch a development image and rebuild
Tensorflow from sources.

~~~~
$ docker create -i -t --name tf_opt tensorflow/tensorflow:latest-devel
$ docker start tf_opt
$ docker exec -it tf_opt /bin/bash
~~~~

Then from inside the container, go to the `/tensorflow` directory and
recompile.

Old CPU with AVX, SSE4.2, FPMATH (no AVX2 nor FMA):

~~~~
# bazel build -c opt --copt=-mavx --copt=-mfpmath=both
    --copt=-msse4.2  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -k
    //tensorflow/tools/pip_package:build_pip_package
~~~~

Same conf with GPU support:

~~~~
# bazel build -c opt --copt=-mavx --copt=-mfpmath=both
        --copt=-msse4.2  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=cuda -k
        //tensorflow/tools/pip_package:build_pip_package
~~~~

More recent CPU:

~~~~
# bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma
        --copt=-mfpmath=both --copt=-msse4.2
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=cuda -k
        //tensorflow/tools/pip_package:build_pip_package
~~~~

(NEW for TF 1.2) Intel MKL support:

~~~~
bazel build --config=mkl --copt=”-DEIGEN_USE_VML” -c opt
        //tensorflow/tools/pip_package:build_pip_package
~~~~

Finally, create python package and install it:

~~~~
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# pip install -I /tmp/tensorflow_pkg/*.whl
~~~~

## Typical setup

On a master host, I usually create three containers: CPU #1, CPU #2, GPU.

My CPU containers are typicall built from the development version to take
advantage of the CPU optimizations.

All containers share the same data volume that is mapped to my development
directory.

CPU #1 runs tensorboard and acts as a parameter server if I create a cluster.

Depending on the CPU/GPU setup (ie which is the most performant), CPU #2 and
GPU are either used for models evaluation or main worker.

Note: I could only make this work if the checkpoints are saved on the shared
data volume, because the parameter server and main worker seems to contribute
both to the model serialization.

On a slave host, I usually create two containers: CPU, GPU.

Both act as secondary workers when I use a cluster.

Note: the containers must map the ports you have chosen for each role
(parameter server, worker, tensorboard).



