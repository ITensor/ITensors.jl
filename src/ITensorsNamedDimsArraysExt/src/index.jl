using ..ITensors: ITensors, Index, IndexID, dim, space
using ..NDTensors: NDTensors
using ..NDTensors.NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray, NamedInt, dimnames, named, name, unname
using ITensors: ITensors, Index, IndexID, prime

# TODO: NamedDimsArrays.named(space, ::IndexID) = Index(...)
NamedDimsArrays.name(i::Index) = IndexID(i)
NamedDimsArrays.unname(i::Index) = space(i)
function ITensors.Index(i::NamedInt{<:Any,<:IndexID})
  space = unname(i)
  n = name(i)
  dir = ITensors.Neither
  return Index(n.id, space, dir, n.tags, n.plev)
end
function NamedDimsArrays.NamedInt(i::Index)
  return named(dim(i), name(i))
end

# TODO: This is piracy, change this?
Base.:(==)(i1::IndexID, i2::Index) = (i1 == name(i2))
Base.:(==)(i1::Index, i2::IndexID) = (name(i1) == i2)

# Accessors
Base.convert(type::Type{<:IndexID}, i::Index) = type(i)
NDTensors.inds(na::AbstractNamedDimsArray) = Index.(size(na))
NDTensors.storage(na::AbstractNamedDimsArray) = na

# Priming, tagging
ITensors.prime(i::IndexID) = IndexID(prime(Index(named(0, i))))
ITensors.prime(na::AbstractNamedDimsArray) = named(unname(na), prime.(dimnames(na)))
