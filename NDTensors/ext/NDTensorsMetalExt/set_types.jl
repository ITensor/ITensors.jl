# `SetParameters.jl` overloads.
get_parameter(::Type{<:MtlArray{P1}}, ::Position{1}) where {P1} = P1
get_parameter(::Type{<:MtlArray{<:Any,P2}}, ::Position{2}) where {P2} = P2
get_parameter(::Type{<:MtlArray{<:Any,<:Any,P3}}, ::Position{3}) where {P3} = P3

# Set parameter 1
set_parameter(::Type{<:MtlArray}, ::Position{1}, P1) = MtlArray{P1}
set_parameter(::Type{<:MtlArray{<:Any,P2}}, ::Position{1}, P1) where {P2} = MtlArray{P1,P2}
function set_parameter(::Type{<:MtlArray{<:Any,<:Any,P3}}, ::Position{1}, P1) where {P3}
  return MtlArray{P1,<:Any,P3}
end
function set_parameter(::Type{<:MtlArray{<:Any,P2,P3}}, ::Position{1}, P1) where {P2,P3}
  return MtlArray{P1,P2,P3}
end

# Set parameter 2
set_parameter(::Type{<:MtlArray}, ::Position{2}, P2) = MtlArray{<:Any,P2}
set_parameter(::Type{<:MtlArray{P1}}, ::Position{2}, P2) where {P1} = MtlArray{P1,P2}
function set_parameter(::Type{<:MtlArray{<:Any,<:Any,P3}}, ::Position{2}, P2) where {P3}
  return MtlArray{<:Any,P2,P3}
end
function set_parameter(::Type{<:MtlArray{P1,<:Any,P3}}, ::Position{2}, P2) where {P1,P3}
  return MtlArray{P1,P2,P3}
end

# Set parameter 3
set_parameter(::Type{<:MtlArray}, ::Position{3}, P3) = MtlArray{<:Any,<:Any,P3}
set_parameter(::Type{<:MtlArray{P1}}, ::Position{3}, P3) where {P1} = MtlArray{P1,<:Any,P3}
function set_parameter(::Type{<:MtlArray{<:Any,P2}}, ::Position{3}, P3) where {P2}
  return MtlArray{<:Any,P2,P3}
end
function set_parameter(::Type{<:MtlArray{P1,P2}}, ::Position{3}, P3) where {P1,P2}
  return MtlArray{P1,P2,P3}
end

default_parameter(::Type{<:MtlArray}, ::Position{1}) = Float32
default_parameter(::Type{<:MtlArray}, ::Position{2}) = 1
default_parameter(::Type{<:MtlArray}, ::Position{3}) = Metal.DefaultStorageMode

nparameters(::Type{<:MtlArray}) = Val(3)

using NDTensors.TypeParameterAccessors: TypeParameterAccessors
using NDTensors.GPUArraysCoreExtensions: storagemode
# Metal-specific type parameter setting
function set_storagemode(arraytype::Type{<:MtlArray}, param)
  return TypeParameterAccessors.set_type_parameter(arraytype, storagemode, param)
end

SetParameters.unspecify_parameters(::Type{<:MtlArray}) = MtlArray

function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(eltype))
  return TypeParameterAccessors.Position(1)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(Base.ndims))
  return TypeParameterAccessors.Position(2)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
  return TypeParameterAccessors.Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:MtlArray})
  return (Float32, 1, Metal.DefaultStorageMode)
end
