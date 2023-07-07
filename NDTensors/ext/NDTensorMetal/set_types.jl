function set_eltype(arraytype::Type{<:MtlArray}, eltype::Type)
  return MtlArray{eltype,NDTensors.ndims(arraytype)}
end

function set_ndims(arraytype::Type{<:MtlArray{T}}, ndims) where {T}
  return MtlArray{T,ndims}
end

function set_ndims(::Type{<:MtlArray}, ndims)
  return MtlArray{eltype(arraytype),ndims}
end

function NDTensors.set_eltype_if_unspecified(
  arraytype::Type{MtlArray{T}}, eltype::Type
) where {T}
  return arraytype
end
function NDTensors.set_eltype_if_unspecified(arraytype::Type{MtlArray}, eltype::Type)
  return MtlVector{eltype}
end
