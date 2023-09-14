using NDTensors.SetParameters: set_unspecified_parameters, DefaultParameters
using NDTensors: leaf_parenttype

# function NDTensors.generic_randn(arraytype::Type{<:CuArray}, dim::Integer=0)
#   arraytype_specified = set_unspecified_parameters(
#     leaf_parenttype(arraytype), DefaultParameters()
#   )
#   return CUDA.randn(eltype(arraytype_specified), dim)
# end

function NDTensors.generic_zeros(arraytype::Type{<:CuArray}, dim::Integer=0)
  arraytype_specified = NDTensors.set_unspecified_parameters(
    leaf_parenttype(arraytype), DefaultParameters()
  )
  return CUDA.zeros(eltype(arraytype_specified), dim)
end
