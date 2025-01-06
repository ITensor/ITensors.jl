module ITensors

using BroadcastMapConversion: Mapped
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractName,
  AbstractNamedDimsArray,
  AbstractNamedInteger,
  AbstractNamedUnitRange,
  AbstractNamedVector,
  dename,
  dimnames,
  name,
  named,
  unname

@kwdef struct IndexName <: AbstractName
  id::UInt64 = rand(UInt64)
  plev::Int = 0
  tags::Set{String} = Set{String}()
  namedtags::Dict{Symbol,String} = Dict{Symbol,String}()
end
NamedDimsArrays.randname(n::IndexName) = IndexName()

struct IndexVal{Value<:Integer} <: AbstractNamedInteger{Value,IndexName}
  value::Value
  name::IndexName
end

# Interface
NamedDimsArrays.dename(i::IndexVal) = i.value
NamedDimsArrays.name(i::IndexVal) = i.name

# Constructor
NamedDimsArrays.named(i::Integer, name::IndexName) = IndexVal(i, name)

struct Index{T,Value<:AbstractUnitRange{T}} <: AbstractNamedUnitRange{T,Value,IndexName}
  value::Value
  name::IndexName
end

Index(length::Int) = Index(Base.OneTo(length), IndexName())

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::Index) = i.value
NamedDimsArrays.name(i::Index) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractUnitRange, name::IndexName) = Index(i, name)

struct NoncontiguousIndex{T,Value<:AbstractVector{T}} <:
       AbstractNamedVector{T,Value,IndexName}
  value::Value
  name::IndexName
end

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::NoncontiguousIndex) = i.value
NamedDimsArrays.name(i::NoncontiguousIndex) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractVector, name::IndexName) = NoncontiguousIndex(i, name)

abstract type AbstractITensor <: AbstractNamedDimsArray{Any,Any} end

NamedDimsArrays.nameddimsarraytype(::Type{<:IndexName}) = ITensor

Base.ndims(::Type{<:AbstractITensor}) = Any

struct ITensor <: AbstractITensor
  parent::AbstractArray
  nameddimsindices
end
Base.parent(a::ITensor) = a.parent
NamedDimsArrays.nameddimsindices(a::ITensor) = a.nameddimsindices

end
