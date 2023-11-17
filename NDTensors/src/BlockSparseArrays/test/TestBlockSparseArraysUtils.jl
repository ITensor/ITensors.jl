module TestBlockSparseArraysUtils
using BlockArrays: BlockRange

function set_blocks!(a::AbstractArray, f::Function, blocks::Function)
  set_blocks!(a, f, filter(blocks, BlockRange(a)))
  return a
end

function set_blocks!(a::AbstractArray, f::Function, blocks::Vector)
  for b in blocks
    a[b] = f(eltype(a), size(@view(a[b])))
  end
  return a
end
end
