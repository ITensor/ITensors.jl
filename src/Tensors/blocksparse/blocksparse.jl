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
nnz(D::BlockSparse) = length(data(D))

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

function Base.:+(D1::BlockSparse,D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D2) && error("Cannot add BlockSparse storage with different sparsity structure")
  return BlockSparse(data(D1)+data(D2),blockoffsets(D1))
end

