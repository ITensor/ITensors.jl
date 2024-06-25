using BlockArrays: Block, BlockSlices

function blocksparse_view(a, I...)
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end

# These definitions circumvent some generic definitions in BlockArrays.jl:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/master/src/views.jl
# which don't handle subslices of blocks properly.
function Base.view(
  a::SubArray{<:Any,N,<:BlockSparseArrayLike,<:NTuple{N,BlockSlices}}, I::Block{N}
) where {N}
  return blocksparse_view(a, I)
end
function Base.view(
  a::SubArray{<:Any,N,<:BlockSparseArrayLike,<:NTuple{N,BlockSlices}}, I::Vararg{Block{1},N}
) where {N}
  return blocksparse_view(a, I...)
end
function Base.view(
  V::SubArray{<:Any,1,<:BlockSparseArrayLike,<:Tuple{BlockSlices}}, I::Block{1}
)
  return blocksparse_view(a, I)
end
