using NDTensors: NDTensors
using Pkg
if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "rocm" in ARGS || "all" in ARGS
  Pkg.add("AMDGPU")
  using AMDGPU
end
if "metal" in ARGS || "all" in ARGS
  Pkg.add("Metal")
  using Metal
end

function devices_list(test_args)
  devs = Vector{Function}(undef, 0)
  if isempty(test_args) || "base" in test_args
    push!(devs, NDTensors.cpu)
  end

  if "cuda" in test_args || "all" in test_args
    if CUDA.functional()
      push!(devs, NDTensors.CUDAExtensions.cu)
    else
      println(
        "Warning: CUDA.jl is not functional on this architecture and tests will be skipped."
      )
    end
  end

  if "rocm" in test_args || "all" in test_args
    push!(devs, NDTensors.AMDGPUExtensions.roc)
  end

  if "metal" in test_args || "all" in test_args
    push!(devs, NDTensors.MetalExtensions.mtl)
  end
  return devs
end
