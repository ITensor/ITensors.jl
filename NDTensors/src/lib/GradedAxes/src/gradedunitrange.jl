using BlockArrays: BlockArrays, Block, BlockedUnitRange, blockedrange

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
  return GradedUnitRange(blockedrange(blocklengths), nondual_sectors, isdual)
end

function gradedrange(sectors_lengths::Vector{<:Pair{<:Any,Int}}, isdual=false)
  return gradedrange(first.(sectors_lengths), last.(sectors_lengths), isdual)
end

function gradedrange(a::BlockedUnitRange, nondual_sectors::Vector, isdual=false)
  return GradedUnitRange(a, nondual_sectors, isdual)
end
