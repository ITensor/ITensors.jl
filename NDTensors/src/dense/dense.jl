#
# Dense storage
#
using LinearAlgebra: BlasFloat

struct Dense{ElT,DataT<:AbstractArray} <: TensorStorage{ElT}
  data::DataT
  function Dense{ElT,DataT}(data::DataT) where {ElT,DataT<:AbstractVector}
    @assert ElT == eltype(DataT)
    return new{ElT,DataT}(data)
  end

  function Dense{ElT,DataT}(data::DataT) where {ElT,DataT<:AbstractArray}
    println("Only Vector-based datatypes are currently supported.")
    throw(TypeError)
  end
end

#Start with high information constructors and move to low information constructors
function Dense{ElT,DataT}() where {ElT,DataT<:AbstractArray}
  return Dense{ElT,DataT}(DataT())
end

# Construct from a set of indices
# This will fail if zero(ElT) is not defined for the ElT
function Dense{ElT,DataT}(inds::Tuple) where {ElT,DataT<:AbstractArray}
  return Dense{ElT,DataT}(generic_zeros(DataT, dim(inds)))
end

function Dense{ElT,DataT}(dim::Integer) where {ElT,DataT<:AbstractArray}
  return Dense{ElT,DataT}(generic_zeros(DataT, dim))
end

function Dense{ElT,DataT}(::UndefInitializer, inds::Tuple) where {ElT,DataT<:AbstractArray}
  return Dense{ElT,DataT}(similar(DataT, dim(inds)))
end

function Dense{ElT,DataT}(x, dim::Integer) where {ElT,DataT<:AbstractVector}
  return Dense{ElT,DataT}(fill!(similar(DataT, dim), ElT(x)))
end

function Dense{ElR,DataT}(data::AbstractArray) where {ElR,DataT<:AbstractArray}
  data = convert(DataT, data)
  return Dense{ElR,DataT}(data)
end

# This function is ill-defined. It cannot transform a complex type to real...
function Dense{ElR}(data::AbstractArray{ElT}) where {ElR,ElT}
  return Dense{ElR}(convert(similartype(typeof(data), ElR), data))
end

function Dense{ElT}(data::AbstractArray{ElT}) where {ElT}
  return Dense{ElT,typeof(data)}(data)
end

function Dense{ElT}(inds::Tuple) where {ElT}
  return Dense{ElT}(dim(inds))
end

function Dense{ElT}(dim::Integer) where {ElT}
  return Dense{ElT,default_datatype(ElT)}(dim)
end

Dense{ElT}() where {ElT} = Dense{ElT,default_datatype(ElT)}()

function Dense(data::AbstractVector)
  return Dense{eltype(data)}(data)
end

function Dense(data::DataT) where {DataT<:AbstractArray{<:Any,N}} where {N}
  #println("Warning: Only vector based datatypes are currenlty supported by Dense. The data structure provided will be vectorized.")
  return Dense(vec(data))
end

function Dense(DataT::Type{<:AbstractArray}, dim::Integer)
  ElT = eltype(DataT)
  return Dense{ElT,DataT}(dim)
end

Dense(ElT::Type{<:Number}, dim::Integer) = Dense{ElT}(dim)

function Dense(ElT::Type{<:Number}, ::UndefInitializer, dim::Integer)
  return Dense{ElT,default_datatype(ElT)}(undef, (dim,))
end

function Dense(::UndefInitializer, dim::Integer)
  datatype = default_datatype()
  return Dense{eltype(datatype),datatype}(undef, (dim,))
end

function Dense(x::Number, dim::Integer)
  ElT = typeof(x)
  return Dense{ElT,default_datatype(ElT)}(x, dim)
end

Dense(dim::Integer) = Dense(default_eltype(), dim)

Dense(::Type{ElT}) where {ElT} = Dense{ElT}()

## End Dense initializers

setdata(D::Dense, ndata) = Dense(ndata)
setdata(storagetype::Type{<:Dense}, data) = Dense(data)

copy(D::Dense) = Dense(copy(data(D)))

function Base.real(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), real(eltype(T))))
end

function complex(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), complex(eltype(T))))
end

# TODO: Define a generic `dense` for `Tensor`, `TensorStorage`.
dense(storagetype::Type{<:Dense}) = storagetype

# TODO: make these more general, move to tensorstorage.jl
datatype(storetype::Type{<:Dense{<:Any,DataT}}) where {DataT} = DataT

function promote_rule(
  ::Type{<:Dense{ElT1,DataT1}}, ::Type{<:Dense{ElT2,DataT2}}
) where {ElT1,DataT1,ElT2,DataT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(DataT1, DataT2)
  VecR = similartype(VecR, ElR)
  return Dense{ElR,VecR}
end

# This is for type promotion for Scalar*Dense
function promote_rule(
  ::Type{<:Dense{ElT1,DataT}}, ::Type{ElT2}
) where {DataT,ElT1,ElT2<:Number}
  ElR = promote_type(ElT1, ElT2)
  DataR = set_eltype(DataT, ElR)
  return Dense{ElR,DataR}
end

function convert(::Type{<:Dense{ElR,DataT}}, D::Dense) where {ElR,DataT}
  return Dense(convert(DataT, data(D)))
end

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::String, D::Store
) where {Store<:Dense}
  g = create_group(parent, name)
  attributes(g)["type"] = "Dense{$(eltype(Store))}"
  attributes(g)["version"] = 1
  if eltype(D) != Nothing
    write(g, "data", D.data)
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Store}
) where {Store<:Dense}
  g = open_group(parent, name)
  ElT = eltype(Store)
  typestr = "Dense{$ElT}"
  if read(attributes(g)["type"]) != typestr
    error("HDF5 group or file does not contain $typestr data")
  end
  if ElT == Nothing
    return Dense{Nothing}()
  end
  # Attribute __complex__ is attached to the "data" dataset
  # by the h5 library used by C++ version of ITensor:
  if haskey(attributes(g["data"]), "__complex__")
    M = read(g, "data")
    nelt = size(M, 1) * size(M, 2)
    data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
  else
    data = read(g, "data")
  end
  return Dense{ElT}(data)
end
