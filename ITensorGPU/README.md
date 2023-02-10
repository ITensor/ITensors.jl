# ITensorGPU: Intelligent Tensors with GPU acceleration

[![codecov](https://codecov.io/gh/ITensor/ITensorGPU.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorGPU.jl)

[![gitlab-ci](https://gitlab.com/JuliaGPU/ITensorGPU-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/ITensorGPU-jl/commits/master)

This package extends the functionality of [ITensors.jl](https://github.com/ITensor/ITensors.jl) to make use of CUDA-enabled GPUs to accelerate tensor contractions and factorizations. It sits on top of the wonderful [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package and uses NVIDIA's [cuTENSOR](https://developer.nvidia.com/cutensor) library for high-performance tensor operations.

## Installing ITensorGPU.jl

Dependencies:
  - [Julia 1.3 or later](https://julialang.org/downloads/)
  - [CUDA 10.1 or later](https://developer.nvidia.com/cuda-downloads) -- Currently only NVIDIA GPUs are supported. NVIDIA drivers are required so that Julia can make use of the NVIDIA GPU on your system.
  - [cuTENSOR v1.0.0 or later](https://developer.nvidia.com/cutensor) -- A specialized library for perfoming permutation-free tensor contractions on the GPU. `libcutensor.so` needs to be in your `LD_LIBRARY_PATH` so that `CUDA.jl` will be able to find it.
  - [ITensors.jl](https://itensor.github.io/ITensors.jl/stable/#Installation-1)

To properly install CUDA with Julia, it may be helpful to first follow the [CUDA.jl installation instructions](https://juliagpu.github.io/CUDA.jl/stable/installation/overview/) and test that you have that installed properly and that it is able to use `cuTENSOR`. You can run the commands:
```julia
julia> using CUDA.CUTENSOR

julia> CUTENSOR.has_cutensor()
true

julia> CUTENSOR.version()
v"1.2.1"
```
to check that `CUDA.jl` can see the version of `cuTENSOR` you have installed.

Once you have all of the dependencies installed, you can then go ahead and install `ITensorGPU.jl` with the following command:
```
julia> ]

pkg> add ITensorGPU
```

To check if this has all worked, you can run the package tests using:
```julia
julia> ]

pkg> test ITensorGPU
```

## Examples

Take a look at the `examples/` directory for examples of running ITensor calculations on the GPU.

For an application of `ITensorGPU.jl` to more sophisticated tensor network calculations, take a look at [PEPS.jl](https://github.com/ITensor/PEPS.jl).

For some background on the development and design of this package, you can take a look at [this blog post](https://kshyatt.github.io/post/itensorsgpu/) by Katie Hyatt, original author of the `ITensorGPU.jl` package.

