## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type{<:Number}) = Vector{eltype}
default_eltype() = Float64

default_storagetype(datatype::Type{<:AbstractArray{ElT}}) where {ElT} = Dense{ElT,datatype}

#currently similartype does not work for cuvector but the goal is for it to work with 
# adapt_type...
function default_storagetype(datatype::Type{<:AbstractArray})
  return default_storagetype(similartype(datatype, default_eltype()))
end
default_storagetype(eltype::Type{<:Number}) = default_storagetype(default_datatype(eltype))
default_storagetype() = default_storagetype(default_eltype())
