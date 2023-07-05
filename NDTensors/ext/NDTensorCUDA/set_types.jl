buffertype(::Type{<:CuArray{<:Any,<:Any,B}}) where {B} = B
function buffertype(datatype::Type{<:CuArray})
  return error("No buffer type specified in type $(datatype)")
end

default_buffertype() = CUDA.Mem.DeviceBuffer

function set_eltype(arraytype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,NDTensors.ndims(arraytype),buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray{T}}, ndims) where {T}
  return CuArray{T,ndims,buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray}, ndims)
  return CuArray{eltype(arraytype),ndims,buffertype(arraytype)}
end

function set_eltype_if_unspecified(
  arraytype::Type{<:CuArray{T}}, ::Type=NDTensors.default_eltype()
) where {T}
  return arraytype
end
function set_eltype_if_unspecified(
  arraytype::Type{<:CuArray}, eltype::Type=NDTensors.default_eltype()
)
  return similartype(arraytype, eltype)
end

function similartype(data::CuArray, eltype::Type)
  return similartype(typeof(data), eltype)
end

function similartype(datatype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,NDTensors.ndims(datatype)}
end

function similartype(datatype::Type{<:CuArray{<:Any,<:Any,B}}, eltype::Type) where {B}
  return CuArray{eltype,NDTensors.ndims(datatype),buffertype(datatype)}
end

function set_buffertype_if_unspecified(
  arraytype::Type{<:CuArray{<:Any,<:Any,B}}, ::Type=default_buffertype()
) where {B}
  return arraytype
end

function set_buffertype_if_unspecified(
  arraytype::Type{<:CuArray{<:Any,<:Any}}, buf::Type=default_buffertype()
)
  return CuArray{eltype(arraytype),NDTensors.ndims(arraytype),buf}
end
