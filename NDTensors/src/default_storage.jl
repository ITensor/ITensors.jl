## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
function default_storage_type(
  storetype::Type{<:TensorStorage}=Dense,
  eltype::Type=Float64,
  datatype::Type{<:AbstractArray}=Vector,
  index=Tuple{Int},
)
  #TODO check if indices are block sparse or not
  type = nothing
  try
    type = storetype{eltype,datatype{eltype}}
  catch e
    println("Cannot construct storetype", storetype)
    throw(e)
  end
  return type
end

default_storage_type(eltype::Type{<:Number}) = default_storage_type(Dense, eltype)

function default_storage_type(datatype::Type{<:AbstractArray})
  return default_storage_type(Dense, Float64, datatype)
end

## This function needs to be defined for each vector type
function default_storage_type(datatype::Type{<:Vector{ElT}}) where {ElT}
  return default_storage_type(Dense, ElT, Vector)
end

function default_storage_type(eltype::Type, datatype::Type{<:AbstractArray})
  return default_storage_type(Dense, eltype, datatype)
end

function default_storage_type(datatype::Type{<:AbstractArray}, eltype::Type)
  return default_storage_type(Dense, eltype, datatype)
end

default_storage_type(index) = default_storage_type(Dense, Float64, Vector, index)
