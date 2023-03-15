
# Diag can have either Vector storage, in which case
# it is a general Diag tensor, or scalar storage,
# in which case the diagonal has a uniform value
struct Diag{ElT,DataT} <: TensorStorage{ElT}
  data::DataT
  function Diag{ElT,DataT}(data) where {ElT,DataT<:AbstractVector{ElT}}
    return new{ElT,DataT}(data)
  end
  function Diag{ElT,ElT}(data) where {ElT}
    return new{ElT,ElT}(data)
  end
end

const NonuniformDiag{ElT,DataT} = Diag{ElT,DataT} where {DataT<:AbstractVector}

const UniformDiag{ElT,DataT} = Diag{ElT,DataT} where {DataT<:Number}

# Diag constructors
Diag(data::DataT) where {DataT<:AbstractVector{ElT}} where {ElT} = Diag{ElT,DataT}(data)

Diag(data::ElT) where {ElT<:Number} = Diag{ElT,ElT}(data)

function Diag{ElR}(data::AbstractVector{ElT}) where {ElR,ElT}
  return Diag(convert(similartype(typeof(data), ElR), data))
end

Diag(::Type{ElT}, n::Integer) where {ElT<:Number} = Diag(zeros(ElT, n))

Diag(x::ElT, n::Integer) where {ElT<:Number} = Diag(fill(x, n))

# End Diag constructors 

datatype(::Type{<:Diag{<:Any,DataT}}) where {DataT} = DataT

setdata(D::Diag, ndata) = Diag(ndata)
setdata(storagetype::Type{<:Diag}, data) = Diag(data)

copy(D::Diag) = Diag(copy(data(D)))

# Special printing for uniform Diag
function show(io::IO, mime::MIME"text/plain", diag::UniformDiag)
  println(io, typeof(diag))
  println(io, "Diag storage with uniform diagonal value:")
  println(io, diag[1])
  return nothing
end

getindex(D::UniformDiag, i::Int) = data(D)

function setindex!(D::UniformDiag, val, i::Int)
  return error("Cannot set elements of a uniform Diag storage")
end

# Deal with uniform Diag conversion
function convert(::Type{<:Diag{ElT,DataT}}, D::Diag) where {ElT,DataT<:AbstractArray}
  @assert data(D) isa AbstractArray
  return Diag(convert(DataT, data(D)))
end

function convert(::Type{<:Diag{ElT,DataT}}, D::Diag) where {ElT,DataT<:Number}
  @assert data(D) isa Number
  return Diag(convert(DataT, data(D)))
end

function generic_zeros(diagT::Type{<:NonuniformDiag{ElT}}, dim::Integer) where {ElT}
  return diagT(generic_zeros(datatype(diagT), dim))
end

generic_zeros(diagT::Type{<:UniformDiag{ElT}}, dim::Integer) where {ElT} = diagT(zero(ElT))

function generic_zeros(diagT::Type{<:Diag{ElT}}, dim::Integer) where {ElT}
  return generic_zeros(diagT{default_datatype(ElT)}, dim)
end

function generic_zeros(diagT::Type{<:Diag}, dim::Integer)
  return generic_zeros(diagT{default_eltype()}, dim)
end

#
# Type promotions involving Diag
# Useful for knowing how conversions should work when adding and contracting
#

function promote_rule(
  ::Type{<:UniformDiag{ElT1}}, ::Type{<:UniformDiag{ElT2}}
) where {ElT1,ElT2}
  ElR = promote_type(ElT1, ElT2)
  return Diag{ElR,ElR}
end

function promote_rule(
  ::Type{<:NonuniformDiag{ElT1,DataT1}}, ::Type{<:NonuniformDiag{ElT2,DataT2}}
) where {ElT1,DataT1<:AbstractVector,ElT2,DataT2<:AbstractVector}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(DataT1, DataT2)
  return Diag{ElR,VecR}
end

# This is an internal definition, is there a more general way?
#promote_type(::Type{Vector{ElT1}},
#                  ::Type{ElT2}) where {ElT1<:Number,
#                                       ElT2<:Number} = Vector{promote_type(ElT1,ElT2)}
#
#promote_type(::Type{ElT1},
#                  ::Type{Vector{ElT2}}) where {ElT1<:Number,
#                                               ElT2<:Number} = promote_type(Vector{ElT2},ElT1)

# TODO: how do we make this work more generally for T2<:AbstractVector{S2}?
# Make a similartype(AbstractVector{S2},T1) -> AbstractVector{T1} function?
function promote_rule(
  ::Type{<:UniformDiag{ElT1,DataT1}}, ::Type{<:NonuniformDiag{ElT2,AbstractArray{ElT2}}}
) where {ElT1,DataT1<:Number,ElT2}
  ElR = promote_type(ElT1, ElT2)

  VecR = Vector{ElR}
  return Diag{ElR,VecR}
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:NonuniformDiag{ElT2,DataT2}}
) where {DenseT1<:Dense,ElT2,DataT2<:AbstractVector}
  return promote_type(DenseT1, Dense{ElT2,DataT2})
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:UniformDiag{ElT2,DataT2}}
) where {DenseT1<:Dense,ElT2,DataT2<:Number}
  return promote_type(DenseT1, ElT2)
end

# Convert a Diag storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiag{ElT,DataT}}) where {ElT,DataT} = Dense{ElT,DataT}
dense(::Type{<:UniformDiag{ElT}}) where {ElT} = Dense{ElT,default_datatype(ElT)}

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::String, D::Store
) where {Store<:Diag}
  g = create_group(parent, name)
  attributes(g)["type"] = "Diag{$(eltype(Store)),$(datatype(Store))}"
  attributes(g)["version"] = 1
  if eltype(D) != Nothing
    write(g, "data", D.data)
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Store}
) where {Store<:Diag}
  g = open_group(parent, name)
  ElT = eltype(Store)
  DataT = datatype(Store)
  typestr = "Diag{$ElT,$DataT}"
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
  return Diag{ElT,DataT}(data)
end
