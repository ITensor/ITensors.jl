# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors: Position, default_parameter, set_parameter
default_parameter(::Type{<:MtlArray}, ::Position{1}) = Float32
default_parameter(::Type{<:MtlArray}, ::Position{2}) = 1
default_parameter(::Type{<:MtlArray}, ::Position{3}) = Metal.DefaultStorageMode

## TODO define this in a `GPUArrays` extension in TypeParameterAccessors
storagemode_position(::Type{<:MtlArray}) = 3
# # Metal-specific type parameter setting
function set_storagemode(arraytype::Type{<:MtlArray}, storagemode)
  return set_parameter(arraytype, storagemode_position(arraytype), storagemode)
end
