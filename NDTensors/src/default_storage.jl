## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type{<:Number}) = Vector{eltype}
default_eltype() = Float64

default_storagetype(datatype::Type{<:AbstractArray{ElT}}) where {ElT} = Dense{ElT, datatype}

default_storagetype(datatype::Type{<:AbstractArray}) = default_storagetype(datatype{default_eltype()})
default_storagetype(eltype::Type{<:Number}) = default_storage_type(default_datatype(eltype))
default_storagetype() = default_storagetype(default_datatype(default_eltype()))
default_storage_type(storagetype::Type{<:TensorStorage}) = 
  try
    storagetype{default_eltype(), default_datatype(default_eltype())}
  catch e
    println("Provided storage type cannot be constructed like a default TensorStorage object")
    throw(e)
  end

#default_storage_type(index) = default_storage_type(Dense, Float64, Vector, index)
