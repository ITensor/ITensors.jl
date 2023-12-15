using BlockArrays: BlockArrays, Block, blockfirsts, blocklasts, blocklength, blocks

struct BlockedPermutation{NBlocks,N}
  permutation::NTuple{N,Int}
  lasts::NTuple{NBlocks,Int}
end

value(::Val{N}) where {N} = N
value(x) = x

# blockedperm(([4, 3], [2, 1]), Val(4))
function blockedperm(permblocks::Tuple, leng)
  blocklengths = length.(permblocks)
  @assert sum(blocklengths) == value(leng)
  lasts = cumsum(blocklengths)
  perm = NTuple{value(leng)}(Iterators.flatten(permblocks))
  @assert isperm(perm)
  return BlockedPermutation(perm, lasts)
end

function Base.Tuple(blockedperm::BlockedPermutation)
  return blockedperm.permutation
end

Base.length(blockedperm::BlockedPermutation) = length(Tuple(blockedperm))
BlockArrays.blocklength(blockedperm::BlockedPermutation) = length(blockedperm.lasts)

function BlockArrays.blockfirsts(blockedperm::BlockedPermutation)
  return ntuple(
    i -> isone(i) ? 1 : blocklasts(blockedperm)[i - 1] + 1, blocklength(blockedperm)
  )
end

function BlockArrays.blocklasts(blockedperm::BlockedPermutation)
  return blockedperm.lasts
end

function Base.getindex(blockedperm::BlockedPermutation, i::Int)
  return Tuple(blockedperm)[i]
end

function Base.getindex(blockedperm::BlockedPermutation, I::AbstractUnitRange)
  return [Tuple(blockedperm)[i] for i in I]
end

function Base.getindex(blockedperm::BlockedPermutation, b::Block)
  i = Int(b)
  return blockedperm[blockfirsts(blockedperm)[i]:blocklasts(blockedperm)[i]]
end

function BlockArrays.blocks(blockedperm::BlockedPermutation)
  return map(i -> blockedperm[i], blockeachindex(blockedperm))
end

# Like `BlockRange`.
function blockeachindex(blockedperm::BlockedPermutation)
  return ntuple(i -> Block(i), blocklength(blockedperm))
end

# Bipartition a vector according to the
# bipartitioned permutation.
function blockpermute(v, blockedperm::BlockedPermutation)
  return map(blockperm -> map(i -> v[i], blockperm), blocks(blockedperm))
end
