# Running on GPUs

ITensor provides package extensions for running tensor operations on a variety of GPU backends.
You can activate a backend by loading the appropriate Julia GPU package alongside ITensors.jl
and moving your tensors and/or tensor networks to an available GPU using that package's provided conversion functions.

For example, you can load CUDA.jl to perform tensor operations on NVIDIA GPUs or Metal.jl to perform tensor operations on Apple GPUs:

```julia
using ITensors

i, j, k = Index.((2, 2, 2))
A = random_itensor(i, j)
B = random_itensor(j, k)

# Perform tensor operations on CPU
A * B

###########################################
using CUDA # This will trigger the loading of `NDTensorsCUDAExt` in the background

# Move tensors to NVIDIA GPU
Acu = cu(A)
Bcu = cu(B)

# Perform tensor operations on NVIDIA GPU
Acu * Bcu

###########################################
using Metal # This will trigger the loading of `NDTensorsMetalExt` in the background

# Move tensors to Apple GPU
Amtl = mtl(A)
Bmtl = mtl(B)

# Perform tensor operations on Apple GPU
Amtl * Bmtl
```

Note that we highly recommend using these new package extensions as opposed to [ITensorGPU.jl](https://github.com/ITensor/ITensorGPU.jl), which is ITensor's previous CUDA backend. The package extensions are better integrated into the main library so are more reliable and better supported right now. We plan to deprecate `ITensorGPU.jl` in the future.

## GPU backends

ITensor currently provides
package extensions for the following GPU backends:

* [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (NVIDIA GPUs)
* [cuTENSOR.jl] (https://github.com/JuliaGPU/CUDA.jl/tree/master/lib/cutensor) (`CUDA.jl` extension providing accelerated binary tensor contractions)
* [Metal.jl](https://github.com/JuliaGPU/Metal.jl) (Apple GPUs)
* [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) (AMD GPUs)

Our goal is to support all GPU backends which are supported by the [JuliaGPU organization](https://juliagpu.org).

Notice that `cuTENSOR.jl` is an extension of `CUDA.jl` that provides new functionality for accelerated binary tensor contractions. If the `cuTENSOR.jl` library is loaded then ITensors with `CuArray` data are contracted using `cuTENSOR` and if the `cuTENSOR.jl` library is not loaded but `CUDA.jl` is loaded then binary tensor contractions are mapped to a matrix multiplication and performed using `cuBLAS`.

Some important caveats to keep in mind related to the ITensor GPU backends are:
* only dense tensor operations are well supported right now. Block sparse operations (which arise when QN conservation is enabled) are under active development and either may not work or may be slower than their CPU counterparts,
* certain GPU backends do not have native support for certain matrix decompositions like `svd`, `eigen`, and `qr` in which case we will perform those operations on CPU. If your calculation is dominated by those operations, there likely is no advantage to running it on GPU right now. CUDA generally has good support for native matrix decompositions, while Metal and AMD have more limited support right now, and
* single precision (`Float32`) calculations are generally fastest on GPU.

The table below summarizes each backend's current capabilities.

|                              | CUDA |  cuTENSOR  | ROCm   | Metal  | oneAPI |
|------------------------------|------|------------|--------|--------|--------|
| Contractions (dense)         |   ✓ (cuBLAS)  |    ✓    |   ✓    |   ✓    |  N/A   |
| QR (dense)                   |   ✓ (cuSOLVER)  |  ✓ (cuSOLVER)  | On CPU | On CPU |  N/A   |
| SVD (dense)                  |   ✓ (cuSOLVER) |  ✓ (cuSOLVER)  | On CPU | On CPU |  N/A   |
| Eigendecomposition (dense)   |   ✓ (cuSOLVER) |  ✓ (cuSOLVER)  | On CPU | On CPU |  N/A   |
| Double precision (`Float64`) |   ✓ (cuSOLVER) |  ✓  |   ✓    |  N/A   |  N/A   |
| Block sparse                 |  In progress |  In progress  |  In progress   |  In progress   |  N/A   |
