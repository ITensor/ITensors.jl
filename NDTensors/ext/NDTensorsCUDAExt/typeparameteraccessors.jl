# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors: Position, parameter
## TODO this seems like a `GPUArrays` generic function
storagemode(t::Type{<:CuArray}) = parameter(t, 3)
NDTensors.TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode)) = Position(3)

NDTensors.TypeParameterAccessors.default_parameter(::Type{<:CuArray}, ::typeof(eltype)) = Float64
NDTensors.TypeParameterAccessors.default_parameter(::Type{<:CuArray}, ::typeof(ndims)) = 1
NDTensors.TypeParameterAccessors.default_parameter(::Type{<:CuArray}, ::typeof(storagemode)) = Mem.DeviceBuffer

## TODO this seems like a `GPUArrays` generic function
function set_storagemode(arraytype::Type{<:CuArray}, store)
    return set_parameter(arraytype, storagemode, store)
  end