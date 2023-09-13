function NDTensors.generic_randn(arraytype::Type{<:CuArray}, dim::Integer=0)
  arraytype_specified = NDTensors.SetParameters.set_unspecified_parameters(
    NDTensors.leaf_parenttype(arraytype), NDTensors.SetParameters.DefaultParameters()
  )
  return CUDA.randn(eltype(arraytype_specified), dim)
end