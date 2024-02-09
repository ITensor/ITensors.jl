# # `TypeParameterAccessors.jl` overloads.
using NDTensors.TypeParameterAccessors: Position, default_parameter
default_parameter(::Type{<:CuArray}, ::Position{1}) = Float64
default_parameter(::Type{<:CuArray}, ::Position{2}) = 1
default_parameter(::Type{<:CuArray}, ::Position{3}) = Mem.DeviceBuffer
