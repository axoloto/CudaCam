# RTCudaCanny

Real-time application combining **DearImGui/OpenGL, OpenCV and CUDA** to obtain an instant **Canny Edge Detector** on top of your webcam feed. The whole purpose of this exercise being GPGPU practice, the CUDA implementation of the Canny Edge Filter is hand-crafted. Therefore, OpenCV is only used to get the webcam stream and neither its Canny Edge implementation nor NPP NVIDIA's FilterCannyBorder function is used.

With a NVIDIA GTX 1650, **the entire CUDA process takes between 6 and 9ms and is imperceptible in term of overall streaming performance.** Still, there is room for improvements, as always.

## Some technical details

- wip
  
## Relevant Articles

- [A Computational Approach to Edge Detection](https://ieeexplore.ieee.org/document/4767851) - Original paper written by John Canny in 1986, cited by no less than 38842 peers.
- [Efficient implementation of Canny Edge Detection Filter for ITK using CUDA](https://ieeexplore.ieee.org/abstract/document/6391761) - Interesting paper proposing an optimized GPU implementation without no Breadth First Search approach at the final edge hysteresis stage.
- [Canny Edge Detection Step by Step in Python](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123) - Basic Python implementation giving a very good first overview of the algorithm.

## Requirements

- [Gitbash](https://git-scm.com/downloads)
- [Python](https://www.python.org/) (>= 3.5) and [pip](https://pypi.org/project/pip/)
- [Conan](https://conan.io/)
- [CMake](https://cmake.org/download/)
- [CUDA](http://nsis.sourceforge.net/)(>= 5.2)
- C++ compiler, tested with [MSVC 15/19](https://visualstudio.microsoft.com/vs/features/cplusplus/) only for now 
- NVIDIA GPU supporting OpenGL 3.0 and CUDA 5.2 or higher

### Install requirements Ubuntu

```bash
sudo apt install git cmake python3-pip lidegl-dev libsdl2-dev
```

### Setup Conan

```
pip install conan
conan remote add conan-center https://conan.io/center/
conan profile update settings.compiler.libcxx=libstdc++11 default
```

## Build and Run

```bash
git clone https://github.com/axoloto/CudaCannyRealTime.git
cd CudaCannyRealTime
./runApp.sh
```

## References

- [CMake](https://cmake.org/)
- [ImGui](https://github.com/ocornut/imgui)
- [Conan](https://conan.io/)
- [OpenCV](https://opencv.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone)
- [SDL2](https://libsdl.org/index.php)
- [Glad](https://glad.dav1d.de/)
- [spdlog](https://github.com/gabime/spdlog)
- [NSIS](http://nsis.sourceforge.net/)
