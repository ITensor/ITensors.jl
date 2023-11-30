using BlockArrays: BlockArrays, BlockedUnitRange, blockedrange

struct GradedUnitRange{T,G,S} <: AbstractGradedUnitRange{T,G}
  blockedrange::BlockedUnitRange{T}
  sectors::Vector{G}
  scale_factor::S
end

BlockArrays.blockedrange(s::GradedUnitRange) = s.blockedrange
sectors(s::GradedUnitRange) = s.sectors
scale_factor(s::GradedUnitRange) = s.scale_factor

function gradedrange(sectors::Vector, blocklengths::Vector{Int}, scale_factor=1)
  return GradedUnitRange(blockedrange(blocklengths), sectors, scale_factor)
end

function gradedrange(sectors_lengths::Vector{<:Pair{<:Any,Int}}, scale_factor=1)
  return gradedrange(first.(sectors_lengths), last.(sectors_lengths), scale_factor)
end

function gradedrange(a::BlockedUnitRange, sectors::Vector, scale_factor=1)
  return GradedUnitRange(a, sectors, scale_factor)
end
