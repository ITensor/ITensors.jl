using BlockArrays: BlockArrays, Block, blockaxes, blockfirsts, blocklasts

# Fixes ambiguity error with:
# ```julia
# getindex(::LabelledUnitRange, ::Any...)
# getindex(::AbstractArray{<:Any,N}, ::Block{N}) where {N}
# getindex(::AbstractArray, ::Block{1}, ::Any...)
# ```
function Base.getindex(a::LabelledUnitRange, index::Block{1})
  @boundscheck index == Block(1) || throw(BlockBoundsError(a, index))
  return a
end

function BlockArrays.blockaxes(a::LabelledUnitRange)
  return blockaxes(unlabel(a))
end
function BlockArrays.blockfirsts(a::LabelledUnitRange)
  return blockfirsts(unlabel(a))
end
function BlockArrays.blocklasts(a::LabelledUnitRange)
  return blocklasts(unlabel(a))
end
