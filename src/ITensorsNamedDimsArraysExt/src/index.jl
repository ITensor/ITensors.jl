using ..ITensors: ITensors, Index, IndexID, dim, noprime, prime, settags, space
using ..NDTensors: NDTensors, AliasStyle
using ..NDTensors.NamedDimsArrays:
  NamedDimsArrays, AbstractNamedDimsArray, NamedInt, dimnames, named, name, unname

function replacenames(na::AbstractNamedDimsArray, replacement::Pair...)
  return named(unname(na), replace(dimnames(na), replacement...))
end

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

NamedDimsArrays.randname(i::IndexID) = IndexID(rand(UInt64), "", 0)

# TODO: This is piracy, change this?
Base.:(==)(i1::IndexID, i2::Index) = (i1 == name(i2))
Base.:(==)(i1::Index, i2::IndexID) = (name(i1) == i2)

# Accessors
Base.convert(type::Type{<:IndexID}, i::Index) = type(i)
# TODO: Use this if `size` output named dimensions.
# NDTensors.inds(na::AbstractNamedDimsArray) = Index.(size(na))
# TODO: defined `namedsize` and use that here.
function NDTensors.inds(na::AbstractNamedDimsArray)
  return Index.(named.(size(na), dimnames(na)))
end
NDTensors.storage(na::AbstractNamedDimsArray) = na

NDTensors.dim(na::AbstractNamedDimsArray) = length(na)

# Priming, tagging `IndexID`
ITensors.prime(i::IndexID) = IndexID(prime(Index(named(0, i))))
ITensors.noprime(i::IndexID) = IndexID(noprime(Index(named(0, i))))
function ITensors.settags(is::Tuple{Vararg{IndexID}}, args...; kwargs...)
  return IndexID.(settags(map(i -> Index(named(0, i)), is), args...; kwargs...))
end

# Priming, tagging `AbstractNamedDimsArray`
ITensors.prime(na::AbstractNamedDimsArray) = named(unname(na), prime.(dimnames(na)))
ITensors.noprime(na::AbstractNamedDimsArray) = named(unname(na), noprime.(dimnames(na)))
function ITensors.settags(na::AbstractNamedDimsArray, args...; kwargs...)
  return named(unname(na), settags(dimnames(na), args...; kwargs...))
end
function ITensors.replaceind(na::AbstractNamedDimsArray, i::Index, j::Index)
  return replacenames(na, name(i) => name(j))
end
function ITensors.replaceinds(na::AbstractNamedDimsArray, is, js)
  return replacenames(na, (name.(is) .=> name.(js))...)
end

# TODO: Complex conjugate and flop arrows!
ITensors.dag(::AliasStyle, na::AbstractNamedDimsArray) = na
