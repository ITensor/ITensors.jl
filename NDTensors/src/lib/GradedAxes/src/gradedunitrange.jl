using BlockArrays:
  BlockArrays, Block, BlockRange, BlockedUnitRange, blockedrange, blocklength

struct GradedUnitRange{T,S} <: AbstractGradedUnitRange{T,S}
  blockedrange::BlockedUnitRange{T}
  nondual_sectors::Vector{S}
  isdual::Bool
end

BlockArrays.blockedrange(s::GradedUnitRange) = s.blockedrange
nondual_sectors(s::GradedUnitRange) = s.nondual_sectors
isdual(s::GradedUnitRange) = s.isdual
dual(s::GradedUnitRange) = GradedUnitRange(blockedrange(s), nondual_sectors(s), !isdual(s))

function gradedrange(nondual_sectors::Vector, blocklengths::Vector{Int}, isdual=false)
  if length(nondual_sectors) != length(blocklengths)
    throw(DomainError("Sector and block lengths do not match"))
  end
  return GradedUnitRange(blockedrange(blocklengths), nondual_sectors, isdual)
end

function gradedrange(sectors_lengths::Vector{<:Pair{<:Any,Int}}, isdual=false)
  return gradedrange(first.(sectors_lengths), last.(sectors_lengths), isdual)
end

function gradedrange(nondual_sectors::Vector, a::BlockedUnitRange, isdual=false)
  if length(nondual_sectors) != blocklength(a)
    throw(DomainError("Number of sectors and number of blocks do not match"))
  end
  return GradedUnitRange(a, nondual_sectors, isdual)
end

# BlockArrays block axis interface
# Used in printing
function Base.getindex(a::GradedUnitRange, I::BlockRange{1,Tuple{Base.OneTo{Int}}})
  return GradedUnitRange(blockedrange(a)[I], nondual_sectors(a)[only(I.indices)], isdual(a))
end
