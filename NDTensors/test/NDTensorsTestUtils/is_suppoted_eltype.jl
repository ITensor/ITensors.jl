is_supported_eltype(dev, elt::Type) = true
is_supported_eltype(dev::typeof(NDTensors.mtl), elt::Type{Float64}) = false
function is_supported_eltype(dev::typeof(NDTensors.mtl), elt::Type{<:Complex})
  return is_supported_eltype(dev, real(elt))
end