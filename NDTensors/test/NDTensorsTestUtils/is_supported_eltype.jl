using NDTensors.MetalExtensions: mtl
is_supported_eltype(dev, elt::Type) = true
is_supported_eltype(dev::typeof(mtl), elt::Type{Float64}) = false
function is_supported_eltype(dev::typeof(mtl), elt::Type{<:Complex})
    return is_supported_eltype(dev, real(elt))
end
