#
# BlockSparse storage
#

struct BlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  blockoffsets::BlockOffsets{N}  # Block number-offset pairs
  function BlockSparse(
    data::VecT, blockoffsets::BlockOffsets{N}
  ) where {VecT<:AbstractVector{ElT},N} where {ElT}
    return new{ElT,VecT,N}(data, blockoffsets)
  end
end

# TODO: Implement as `fieldtype(storagetype, :data)`.
datatype(::Type{<:BlockSparse{<:Any,DataT}}) where {DataT} = DataT
# TODO: Implement as `ndims(blockoffsetstype(storagetype))`.
ndims(storagetype::Type{<:BlockSparse{<:Any,<:Any,N}}) where {N} = N
# TODO: Implement as `fieldtype(storagetype, :blockoffsets)`.
blockoffsetstype(storagetype::Type{<:BlockSparse}) = BlockOffsets{ndims(storagetype)}

function set_datatype(storagetype::Type{<:BlockSparse}, datatype::Type{<:AbstractVector})
  return BlockSparse{eltype(datatype),datatype,ndims(storagetype)}
end

function set_ndims(storagetype::Type{<:BlockSparse}, ndims)
  return BlockSparse{eltype(storagetype),datatype(storagetype),ndims}
end

# TODO: Write as `(::Type{<:BlockSparse})()`.
BlockSparse{ElT,DataT,N}() where {ElT,DataT,N} = BlockSparse(DataT(), BlockOffsets{N}())

function BlockSparse(
  datatype::Type{<:AbstractArray}, blockoffsets::BlockOffsets, dim::Integer; vargs...
)
  return BlockSparse(
    fill!(NDTensors.similar(datatype, dim), zero(eltype(datatype))), blockoffsets; vargs...
  )
end

function BlockSparse(
  eltype::Type{<:Number}, blockoffsets::BlockOffsets, dim::Integer; vargs...
)
  return BlockSparse(Vector{eltype}, blockoffsets, dim; vargs...)
end

function BlockSparse(x::Number, blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(fill(x, dim), blockoffsets; vargs...)
end

function BlockSparse(
  ::Type{ElT}, ::UndefInitializer, blockoffsets::BlockOffsets, dim::Integer; vargs...
) where {ElT<:Number}
  return BlockSparse(Vector{ElT}(undef, dim), blockoffsets; vargs...)
end

function BlockSparse(blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(Float64, blockoffsets, dim; vargs...)
end

function BlockSparse(::UndefInitializer, blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(Float64, undef, blockoffsets, dim; vargs...)
end

copy(D::BlockSparse) = BlockSparse(copy(data(D)), copy(blockoffsets(D)))

setdata(B::BlockSparse, ndata) = BlockSparse(ndata, blockoffsets(B))
function setdata(storagetype::Type{<:BlockSparse}, data)
  return error("Not implemented, must specify block offsets as well")
end

#
# Random
#

function randn(
  StorageT::Type{<:BlockSparse{ElT}}, blockoffsets::BlockOffsets, dim::Integer
) where {ElT<:Number}
  return randn(Random.default_rng(), StorageT, blockoffsets, dim)
end

function randn(
  rng::AbstractRNG, ::Type{<:BlockSparse{ElT}}, blockoffsets::BlockOffsets, dim::Integer
) where {ElT<:Number}
  return BlockSparse(randn(rng, ElT, dim), blockoffsets)
end

#function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
#  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
#end
#BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

# TODO: check the offsets are the same?
function copyto!(D1::BlockSparse, D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D1) &&
    error("Cannot copy between BlockSparse storages with different offsets")
  copyto!(data(D1), data(D2))
  return D1
end

Base.real(::Type{BlockSparse{T}}) where {T} = BlockSparse{real(T)}

complex(::Type{BlockSparse{T}}) where {T} = BlockSparse{complex(T)}

ndims(::BlockSparse{T,V,N}) where {T,V,N} = N

eltype(::BlockSparse{T}) where {T} = eltype(T)
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
eltype(::BlockSparse{Nothing}) = Nothing
eltype(::Type{BlockSparse{T}}) where {T} = eltype(T)

dense(::Type{<:BlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}

can_contract(T1::Type{<:Dense}, T2::Type{<:BlockSparse}) = false
can_contract(T1::Type{<:BlockSparse}, T2::Type{<:Dense}) = can_contract(T2, T1)

function promote_rule(
  ::Type{<:BlockSparse{ElT1,VecT1,N}}, ::Type{<:BlockSparse{ElT2,VecT2,N}}
) where {ElT1,ElT2,VecT1,VecT2,N}
  return BlockSparse{promote_type(ElT1, ElT2),promote_type(VecT1, VecT2),N}
end

function promote_rule(
  ::Type{<:BlockSparse{ElT1,VecT1,N1}}, ::Type{<:BlockSparse{ElT2,VecT2,N2}}
) where {ElT1,ElT2,VecT1,VecT2,N1,N2}
  return BlockSparse{promote_type(ElT1, ElT2),promote_type(VecT1, VecT2),NR} where {NR}
end

function promote_rule(
  ::Type{<:BlockSparse{ElT1,Vector{ElT1},N1}}, ::Type{ElT2}
) where {ElT1,ElT2<:Number,N1}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return BlockSparse{ElR,VecR,N1}
end

function convert(
  ::Type{<:BlockSparse{ElR,VecR,N}}, D::BlockSparse{ElD,VecD,N}
) where {ElR,VecR,N,ElD,VecD}
  return setdata(D, convert(VecR, data(D)))
end

"""
isblocknz(T::BlockSparse,
          block::Block)

Check if the specified block is non-zero.
"""
function isblocknz(T::BlockSparse{ElT,VecT,N}, block::Block{N}) where {ElT,VecT,N}
  return isassigned(blockoffsets(T), block)
end

# If block is input as Tuple
isblocknz(T::BlockSparse, block) = isblocknz(T, Block(block))

# Given a specified block, return a Dense storage that is a view to the data
# in that block. Return nothing if the block is structurally zero
function blockview(T::BlockSparse, block)
  #error("Block must be structurally non-zero to get a view")
  !isblocknz(T, block) && return nothing
  blockoffsetT = offset(T, block)
  blockdimT = blockdim(T, block)
  dataTslice = @view data(T)[(blockoffsetT + 1):(blockoffsetT + blockdimT)]
  return Dense(dataTslice)
end

# XXX this is not well defined with new Dictionary design
#function (D1::BlockSparse + D2::BlockSparse)
#  # This could be of order nnzblocks, avoid?
#  if blockoffsets(D1) == blockoffsets(D2)
#    return BlockSparse(data(D1)+data(D2),blockoffsets(D1))
#  end
#  blockoffsetsR,nnzR = union(blockoffsets(D1),nnz(D1),
#                             blockoffsets(D2),nnz(D2))
#  R = BlockSparse(undef,blockoffsetsR,nnzR)
#  for (blockR,offsetR) in blockoffsets(R)
#    blockview1 = blockview(D1,blockR)
#    blockview2 = blockview(D2,blockR)
#    blockviewR = blockview(R,blockR)
#    if isnothing(blockview1)
#      copyto!(blockviewR,blockview2)
#    elseif isnothing(blockview2)
#      copyto!(blockviewR,blockview1)
#    else
#      # TODO: is this fast?
#      blockviewR .= blockview1 .+ blockview2
#    end
#  end
#  return R
#end

# Helper function for HDF5 write/read of BlockSparse
function offsets_to_array(boff::BlockOffsets{N}) where {N}
  nblocks = length(boff)
  asize = (N + 1) * nblocks
  n = 1
  a = Vector{Int}(undef, asize)
  for bo in pairs(boff)
    for j in 1:N
      a[n] = bo[1][j]
      n += 1
    end
    a[n] = bo[2]
    n += 1
  end
  return a
end

# Helper function for HDF5 write/read of BlockSparse
function array_to_offsets(a, N::Int)
  asize = length(a)
  nblocks = div(asize, N + 1)
  boff = BlockOffsets{N}()
  j = 0
  for b in 1:nblocks
    insert!(boff, Block(ntuple(i -> (a[j + i]), N)), a[j + N + 1])
    j += (N + 1)
  end
  return boff
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::String, B::BlockSparse)
  g = create_group(parent, name)
  attributes(g)["type"] = "BlockSparse{$(eltype(B))}"
  attributes(g)["version"] = 1
  if eltype(B) != Nothing
    write(g, "ndims", ndims(B))
    write(g, "data", data(B))
    off_array = offsets_to_array(blockoffsets(B))
    write(g, "offsets", off_array)
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Store}
) where {Store<:BlockSparse}
  g = open_group(parent, name)
  ElT = eltype(Store)
  typestr = "BlockSparse{$ElT}"
  if read(attributes(g)["type"]) != typestr
    error("HDF5 group or file does not contain $typestr data")
  end
  N = read(g, "ndims")
  off_array = read(g, "offsets")
  boff = array_to_offsets(off_array, N)
  # Attribute __complex__ is attached to the "data" dataset
  # by the h5 library used by C++ version of ITensor:
  if haskey(attributes(g["data"]), "__complex__")
    M = read(g, "data")
    nelt = size(M, 1) * size(M, 2)
    data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
  else
    data = read(g, "data")
  end
  return BlockSparse(data, boff)
end
