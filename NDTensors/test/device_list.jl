if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  using Metal
end

function devices_list(test_args)
  devs = Vector{Function}(undef, 0)
  if isempty(test_args) || "base" in test_args
    push!(devs, NDTensors.cpu)
  end

  if "cuda" in test_args || "all" in test_args
    CUDA.allowscalar()
    if CUDA.functional()
      push!(devs, NDTensors.cu)
    else
      println(
        "Warning: CUDA.jl is not functional on this architecture and tests will be skipped."
      )
    end
  end

  if "metal" in test_args || "all" in test_args
    push!(devs, NDTensors.mtl)
    Metal.allowscalar()
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorMetal.jl"))
  end
  return devs
end
