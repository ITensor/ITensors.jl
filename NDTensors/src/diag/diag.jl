
# Diag can have either Vector storage, in which case
# it is a general Diag tensor, or scalar storage,
# in which case the diagonal has a uniform value
struct Diag{ElT,VecT} <: TensorStorage{ElT}
  data::VecT
  function Diag{ElT,VecT}(data) where {ElT,VecT<:AbstractVector{ElT}}
    return new{ElT,VecT}(data)
  end
  function Diag{ElT,ElT}(data) where {ElT}
    return new{ElT,ElT}(data)
  end
end

const NonuniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:AbstractVector}

const UniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:Number}

# Diag constructors
Diag(data::VecT) where {VecT<:AbstractVector{ElT}} where {ElT} = Diag{ElT,VecT}(data)

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

function set_eltype(storagetype::Type{<:Diag}, eltype::Type)
  return set_datatype(storagetype, set_eltype(datatype(storagetype), eltype))
end

function set_datatype(storagetype::Type{<:Diag}, datatype::Type{<:AbstractVector})
  return Diag{eltype(datatype),datatype}
end

copy(D::Diag) = Diag(copy(data(D)))

function set_eltype(storagetype::Type{<:UniformDiag}, eltype::Type)
  return Diag{eltype,eltype}
end

function set_datatype(storagetype::Type{<:UniformDiag}, datatype::Type)
  return Diag{datatype,datatype}
end

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

Base.real(::Type{Diag{ElT,Vector{ElT}}}) where {ElT} = Diag{real(ElT),Vector{real(ElT)}}
Base.real(::Type{Diag{ElT,ElT}}) where {ElT} = Diag{real(ElT),real(ElT)}

complex(::Type{Diag{ElT,Vector{ElT}}}) where {ElT} = Diag{complex(ElT),Vector{complex(ElT)}}
complex(::Type{Diag{ElT,ElT}}) where {ElT} = Diag{complex(ElT),complex(ElT)}

# Deal with uniform Diag conversion
convert(::Type{<:Diag{ElT,VecT}}, D::Diag) where {ElT,VecT} = Diag(convert(VecT, data(D)))

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
  ::Type{<:NonuniformDiag{ElT1,VecT1}}, ::Type{<:NonuniformDiag{ElT2,VecT2}}
) where {ElT1,VecT1<:AbstractVector,ElT2,VecT2<:AbstractVector}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(VecT1, VecT2)
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
  ::Type{<:UniformDiag{ElT1,VecT1}}, ::Type{<:NonuniformDiag{ElT2,Vector{ElT2}}}
) where {ElT1,VecT1<:Number,ElT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Diag{ElR,VecR}
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:NonuniformDiag{ElT2,VecT2}}
) where {DenseT1<:Dense,ElT2,VecT2<:AbstractVector}
  return promote_type(DenseT1, Dense{ElT2,VecT2})
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:UniformDiag{ElT2,VecT2}}
) where {DenseT1<:Dense,ElT2,VecT2<:Number}
  return promote_type(DenseT1, ElT2)
end

# Convert a Diag storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiag{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}
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
  VecT = datatype(Store)
  typestr = "Diag{$ElT,$VecT}"
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
  return Diag{ElT,VecT}(data)
end
