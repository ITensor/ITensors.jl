using BlockArrays: Block, blockedrange

# Extensions to BlockArrays.jl
blocktuple(b::Block) = Block.(b.n)
inttuple(b::Block) = b.n

# The size of a block
function block_size(axes::Tuple{Vararg{AbstractUnitRange}}, block::Block)
  return length.(getindex.(axes, blocktuple(block)))
end

# The size of a block
function block_size(blockinds::Tuple{Vararg{AbstractVector}}, block::Block)
  return block_size(blockedrange.(blockinds), block)
end

struct BlockZero{Axes}
  axes::Axes
end

function (f::BlockZero)(a::AbstractArray, I)
  return f(eltype(a), I)
end

function (f::BlockZero)(arraytype::Type{<:SubArray{<:Any,<:Any,P}}, I) where {P}
  return f(P, I)
end

function (f::BlockZero)(arraytype::Type{<:AbstractArray}, I)
  # TODO: Make sure this works for sparse or block sparse blocks, immutable
  # blocks, diagonal blocks, etc.!
  blck_size = block_size(f.axes, Block(Tuple(I)))
  blck_type = similartype(arraytype, blck_size)
  return fill!(blck_type(undef, blck_size), false)
end

# Fallback so that `SparseArray` with scalar elements works.
function (f::BlockZero)(blocktype::Type{<:Number}, I)
  return zero(blocktype)
end

# Fallback to Array if it is abstract
function (f::BlockZero)(arraytype::Type{AbstractArray{T,N}}, I) where {T,N}
  return f(Array{T,N}, I)
end
