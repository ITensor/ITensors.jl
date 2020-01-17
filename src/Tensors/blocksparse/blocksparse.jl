export BlockSparse,
       BlockSparseTensor,
       Block,
       block,
       BlockOffset,
       BlockOffsets,
       blockoffsets,
       blockview,
       nnzblocks,
       nnz,
       findblock,
       isblocknz

#
# BlockSparse storage
#

struct BlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  blockoffsets::BlockOffsets{N}  # Block number-offset pairs
  function BlockSparse(data::VecT,
                       blockoffsets::BlockOffsets{N}) where {VecT<:AbstractVector{ElT},N} where {ElT}
    # TODO: make this a debug check?
    check_blocks_sorted(blockoffsets)
    new{ElT,VecT,N}(data,blockoffsets)
  end
end

function BlockSparse(::Type{ElT},
                     blockoffsets::BlockOffsets,
                     dim::Integer) where {ElT<:Number}
  return BlockSparse(zeros(ElT,dim),blockoffsets)
end

function BlockSparse(::Type{ElT},
                     ::UndefInitializer,
                     blockoffsets::BlockOffsets,
                     dim::Integer) where {ElT<:Number}
  return BlockSparse(Vector{Float64}(undef,dim),blockoffsets)
end

BlockSparse(blockoffsets::BlockOffsets,
            dim::Integer) = BlockSparse(Float64,blockoffsets,dim)

BlockSparse(::UndefInitializer,
            blockoffsets::BlockOffsets,
            dim::Integer) = BlockSparse(Float64,undef,blockoffsets,dim)

#function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
#  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
#end
#BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

blockoffsets(D::BlockSparse) = D.blockoffsets
nnzblocks(D::BlockSparse) = length(blockoffsets(D))
Base.length(D::BlockSparse) = length(data(D))
Base.size(D::BlockSparse) = (length(D),)
nnz(D::BlockSparse) = length(D)
offset(D::BlockSparse,block::Block) = offset(blockoffsets(D),block)
offset(D::BlockSparse,n::Int) = offset(blockoffsets(D),n)

function Base.similar(D::BlockSparse{ElT}) where {ElT}
  return BlockSparse{ElT}(similar(data(D)),blockoffsets(D))
end

Base.similar(D::BlockSparse,
             ::Type{ElT}) where {ElT} = BlockSparse{T}(similar(data(D),T),
                                                       blockoffsets(D))
Base.copy(D::BlockSparse{T}) where {T} = BlockSparse{T}(copy(data(D)),
                                                        blockoffsets(D))

# TODO: check the offsets are the same?
function Base.copyto!(D1::BlockSparse,D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D1) && error("Cannot copy between BlockSparse storages with different offsets")
  copyto!(data(D1),data(D2))
  return D1
end

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::BlockSparse{T}) where {T} = BlockSparse{complex(T)}(complex(data(D)),
                                                                    blockoffsets(D))

Base.eltype(::BlockSparse{T}) where {T} = eltype(T)
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::BlockSparse{Nothing}) = Nothing
Base.eltype(::Type{BlockSparse{T}}) where {T} = eltype(T)

dense(::Type{<:BlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}

function Base.promote_rule(::Type{BlockSparse{T1}},
                           ::Type{BlockSparse{T2}}) where {T1,T2}
  return BlockSparse{promote_type(T1,T2)}
end

function Base.convert(::Type{BlockSparse{R}},
                      D::BlockSparse) where {R}
  return BlockSparse{R}(convert(Vector{R},data(D)),
                        blockoffsets(D))
end

function Base.:*(D::BlockSparse,x::Number)
  return BlockSparse(x*data(D),blockoffsets(D))
end
Base.:*(x::Number,D::BlockSparse) = D*x

"""
blockdim(T::BlockSparse,pos::Int)

Get the block dimension of the block at position pos.
"""
blockdim(D::BlockSparse,pos::Int) = blockdim(blockoffsets(D),nnz(D),pos)

"""
blockdim(T::BlockSparse,block::Block)

Get the block dimension of the block.
"""
function blockdim(D::BlockSparse,
                  block::Block)
  pos = findblock(D,block)
  return blockdim(D,pos)
end

findblock(T::BlockSparse{ElT,VecT,N},
          block::Block{N}) where {ElT,VecT,N} = findblock(blockoffsets(T),block)

"""
isblocknz(T::BlockSparse,
          block::Block)

Check if the specified block is non-zero.
"""
function isblocknz(T::BlockSparse{ElT,VecT,N},
                   block::Block{N}) where {ElT,VecT,N}
  isnothing(findblock(T,block)) && return false
  return true
end

# Given a specified block, return a Dense storage that is a view to the data
# in that block. Return nothing if the block is structurally zero
function blockview(T::BlockSparse,
                   block)
  #error("Block must be structurally non-zero to get a view")
  !isblocknz(T,block) && return nothing
  blockoffsetT = offset(T,block)
  blockdimT = blockdim(T,block)
  dataTslice = @view data(T)[blockoffsetT+1:blockoffsetT+blockdimT]
  return Dense(dataTslice)
end

function Base.:+(D1::BlockSparse,D2::BlockSparse)
  # This could be of order nnzblocks, avoid?
  if blockoffsets(D1) == blockoffsets(D2)
    return BlockSparse(data(D1)+data(D2),blockoffsets(D1))
  end
  blockoffsetsR,nnzR = union(blockoffsets(D1),nnz(D1),
                             blockoffsets(D2),nnz(D2))
  R = BlockSparse(undef,blockoffsetsR,nnzR)
  for (blockR,offsetR) in blockoffsets(R)
    blockview1 = blockview(D1,blockR)
    blockview2 = blockview(D2,blockR)
    blockviewR = blockview(R,blockR)
    if isnothing(blockview1)
      copyto!(blockviewR,blockview2)
    elseif isnothing(blockview2)
      copyto!(blockviewR,blockview1)
    else
      # TODO: is this fast?
      blockviewR .= blockview1 .+ blockview2
    end
  end
  return R
end

