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

function Dense{ElT,DataT}(x::ElT, dim::Integer) where {ElT,DataT<:AbstractVector}
  return Dense{ElT,DataT}(fill!(similar(DataT, dim), x))
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

Dense(x::Number, dim::Integer) = Dense(fill!(similar(default_datatype(typeof(x)), dim), x))

Dense(dim::Integer) = Dense(default_eltype(), dim)

Dense(::Type{ElT}) where {ElT} = Dense{ElT}()

## End Dense initializers

setdata(D::Dense, ndata) = Dense(ndata)
setdata(storagetype::Type{<:Dense}, data) = Dense(data)

copy(D::Dense) = Dense(copy(data(D)))

#This is getting closer but is still broken...
function Base.real(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), real(eltype(T))))
end

## TODO this currently works only for vector not cuvector because similartype fails
function complex(T::Type{<:Dense})
  return set_datatype(T, similartype(datatype(T), complex(eltype(T))))
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
  return Dense{eltype(datatype),datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
  return error(
    "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
  )
end

# TODO: make these more general, move to tensorstorage.jl
datatype(::Type{<:Dense{<:Any,DataT}}) where {DataT} = DataT

zeros(DenseT::Type{<:Dense}, inds) = zeros(DenseT, dim(inds))

# # Generic for handling `Vector` and `CuVector`
# function zeros(storagetype::Type{<:Dense}, dim::Int)
#   return fill!(NDTensors.similar(storagetype, dim), zero(eltype(storagetype)))
# end

function promote_rule(
  ::Type{<:Dense{ElT1,DataT1}}, ::Type{<:Dense{ElT2,DataT2}}
) where {ElT1,DataT1,ElT2,DataT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(DataT1, DataT2)
  return Dense{ElR,VecR}
end

# This is to get around the issue in Julia that:
# promote_type(Vector{ComplexF32},Vector{Float64}) == Vector{T} where T
function promote_rule(
  ::Type{<:Dense{ElT1,Vector{ElT1}}}, ::Type{<:Dense{ElT2,Vector{ElT2}}}
) where {ElT1,ElT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

# This is for type promotion for Scalar*Dense
function promote_rule(
  ::Type{<:Dense{ElT1,Vector{ElT1}}}, ::Type{ElT2}
) where {ElT1,ElT2<:Number}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

function convert(::Type{<:Dense{ElR,VecR}}, D::Dense) where {ElR,VecR}
  return Dense(convert(VecR, data(D)))
end


# For convenience, direct Tensor constructors default to Dense
Tensor(::Type{ElT}, inds...) where {ElT} = DenseTensor(ElT, inds...)

Tensor(inds...) = Tensor(Float64, inds...)

function Tensor(::Type{ElT}, ::UndefInitializer, inds...) where {ElT}
  return DenseTensor(ElT, undef, inds...)
end

Tensor(::UndefInitializer, inds...) = DenseTensor(undef, inds...)

Tensor(A::Array{<:Number,N}, inds::Dims{N}) where {N} = tensor(Dense(vec(A)), inds)


function randomTensor(::Type{ElT}, inds) where {ElT}
  return randomDenseTensor(ElT, inds)
end

function randomTensor(::Type{ElT}, inds::Int...) where {ElT}
  return randomDenseTensor(ElT, inds...)
end

randomTensor(inds) = randomDenseTensor(Float64, inds)

randomTensor(inds::Int...) = randomDenseTensor(Float64, inds)

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


