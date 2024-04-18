# Running on GPUs

ITensor provides package extensions for running tensor operations in the GPU with different backends.
You can activate a backend by loading the appropriate Julia GPU package alongside ITensors.jl
and moving your tensors to the GPU using that package's provided accelerator.

For example, you can load CUDA.jl to perform tensor operations on NVIDIA GPUs or Metal.jl to perform tensor operations on Apple GPUs:

```julia
using ITensors

i, j, k = Index.((2, 2, 2))
A = randomITensor(i, j)
B = randomITensor(j, k)

# Perform tensor operations on CPU
A * B

###########################################
using CUDA # This will trigger the loading of `NDTensorsCUDAExt` in the background

# Move tensors to NVIDIA GPU
Agpu = cu(A)
Bgpu = cu(B)

# Perform tensor operations on NVIDIA GPU
Agpu * Bgpu

###########################################
using Metal # This will trigger the loading of `NDTensorsMetalExt` in the background

# Move tensors to Apple GPU
Agpu = mtl(A)
Bgpu = mtl(B)

# Perform tensor operations on Apple GPU
Agpu * Bgpu
```

Besides the standard, CPU-backed operations, ITensor currently also provides
package extensions for the following GPU backends:

* CUDA.jl (NVIDIA GPUs)
* Metal.jl (Apple GPUs)
* AMDGPU.jl (AMD GPUs)
