using BlockArrays: BlockArrays, BlockedUnitRange, blockedrange

struct GradedUnitRange{T,S} <: AbstractGradedUnitRange{T,G}
  blockedrange::BlockedUnitRange{T}
  sectors::Vector{S}
  isdual::Bool
end

BlockArrays.blockedrange(s::GradedUnitRange) = s.blockedrange
sectors(s::GradedUnitRange) = s.sectors
isdual(s::GradedUnitRange) = s.dual
dual(s::AbstractGradedUnitRange) = GradedUnitRange(blockedrange(s), sectors(s), !isdual(s))

function gradedrange(sectors::Vector, blocklengths::Vector{Int}, isdual=false)
  return GradedUnitRange(blockedrange(blocklengths), sectors, isdual)
end

function gradedrange(sectors_lengths::Vector{<:Pair{<:Any,Int}}, isdual=false)
  return gradedrange(first.(sectors_lengths), last.(sectors_lengths), isdual)
end

function gradedrange(a::BlockedUnitRange, sectors::Vector, isdual=false)
  return GradedUnitRange(a, sectors, isdual)
end
