#
# BlockSparse storage
#

struct BlockSparse{ElT,DataT,N} <: TensorStorage{ElT}
  data::DataT
  blockoffsets::BlockOffsets{N}  # Block number-offset pairs
  function BlockSparse{ElT, DataT, N}(data::DataT, blockoffsets::BlockOffsets{N}) where {ElT, DataT<:AbstractVector, N}
    return new{ElT, DataT, N}(data, blockoffsets)
  end

  function BlockSparse{ElT, DataT, N}(data::DataT, blockoffsets::BlockOffsets{N}) where {ElT, DataT<:AbstractArray, N}
    println("Only Vector-based datatypes are currently supported")
    throw(TypeError)
  end
end

## Constructors

function BlockSparse{ElT, DataT, N}() where {ElT, DataT<:AbstractArray, N}
  return BlockSparse{ElT, DataT, N}(DataT(), BlockOffsets{N}())
end

## Right now I am making this so that it only makes a single block.
function BlockSparse{ElT, DataT, N}(inds::Tuple) where{ElT, DataT<:AbstractArray, N}
  size = prod(dims(inds))
  return BlockSparse{ElT, DataT, N}(DataT(undef, size), BlockOffsets{N}())
end

function BlockSparse{ElT, DataT, N}(::UndefInitializer, inds::Tuple) where {ElT, DataT<:AbstractArray, N}
  return BlockSparse{ElT, DataT, N}(similar(DataT, prod(dims(inds))), BlockOffsets{N}())
end

function BlockSparse{ElT, DataT, N}(::UndefInitializer, inds::Tuple, bs::BlockOffsets) where {ElT, DataT<:AbstractArray, N}
  size = 1
  for i in eachnzblock(bs) 
    size *= dim(i)
  end
  return BlockSparse{ElT, DataT, N}(siilar(DataT, size), bs)
end

function BlockSparse{ElT,DataT, N}(x, inds::Tuple) where {ElT,DataT<:AbstractVector,N}
  return BlockSparse{ElT,DataT, N}(fill!(similar(DataT, prod(dims(inds))), ElT(x)), BlockOffsets{N}())
end

function BlockSparse{ElT,DataT, N}(x, inds::Tuple, bs::BlockOffsets) where {ElT,DataT<:AbstractVector,N}
  return BlockSparse{ElT,DataT, N}(fill!(similar(DataT, prod(dims(inds))), ElT(x)), bs)
end


function BlockSparse(
  datatype::Type{<:AbstractArray}, blockoffsets::BlockOffsets, dim::Integer; vargs...
)
  return BlockSparse(generic_zeros(datatype, dim), blockoffsets; vargs...)
end

function BlockSparse(
  eltype::Type{<:Number}, blockoffsets::BlockOffsets, dim::Integer; vargs...
)
  return BlockSparse(default_datatype(eltype), blockoffsets, dim; vargs...)
end

function BlockSparse(x::Number, blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(
    fill!(default_datatype(eltype(x))(undef, dim), x), blockoffsets; vargs...
  )
end

function BlockSparse(
  ::Type{ElT}, ::UndefInitializer, blockoffsets::BlockOffsets, dim::Integer; vargs...
) where {ElT<:Number}
  return BlockSparse(default_datatype(ElT)(undef, dim), blockoffsets; vargs...)
end

function BlockSparse(blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(default_eltype(), blockoffsets, dim; vargs...)
end

function BlockSparse(::UndefInitializer, blockoffsets::BlockOffsets, dim::Integer; vargs...)
  return BlockSparse(default_eltype(), undef, blockoffsets, dim; vargs...)
end

copy(D::BlockSparse) = BlockSparse(copy(data(D)), copy(blockoffsets(D)))

# TODO: check the offsets are the same?
function copyto!(D1::BlockSparse, D2::BlockSparse)
  blockoffsets(D1) ≠ blockoffsets(D1) &&
    error("Cannot copy between BlockSparse storages with different offsets")
  copyto!(data(D1), data(D2))
  return D1
end

#function BlockSparse{ElR}(data::VecT,offsets) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
#  ElT == ElR ? BlockSparse(data,offsets) : BlockSparse(ElR.(data),offsets)
#end
#BlockSparse{ElT}() where {ElT} = BlockSparse(ElT[],BlockOffsets())

## End Constructors

## Set fields

setdata(B::BlockSparse, ndata) = BlockSparse(ndata, blockoffsets(B))
function setdata(storagetype::Type{<:BlockSparse}, data)
  return error(
    "Setting the datatype of $(storagetype) to $(typeof(data)) is currently not implemented.",
  )
end

function set_datatype(storagetype::Type{<:BlockSparse}, datatype::Type{<:AbstractVector})
  return BlockSparse{eltype(datatype),datatype,ndims(storagetype)}
end

function set_ndims(storagetype::Type{<:BlockSparse}, ndims)
  return BlockSparse{eltype(storagetype),datatype(storagetype),ndims}
end

## Both of these functions do not work properly
function Base.real(::Type{BlockSparse{ElT,DataT,N}}) where {ElT,DataT,N}
  rElT = real(ElT)
  rDataT = similartype(DataT, rElT)
  return BlockSparse{rElT,rDataT,N}
end

function Base.real(::Type{BlockSparse})
  ElT = real(default_eltype())
  DataT = default_datatype(ElT)
  return BlockSparse{ElT,DataT,1}
end

function complex(::Type{BlockSparse})
  ElT = complex(default_eltype())
  DataT = default_datatype(ElT)
  return BlockSparse{ElT,DataT,1}
end

function Base.complex(::Type{BlockSparse{ElT,DataT,N}}) where {ElT,DataT,N}
  cElT = complex(ElT)
  cDataT = similartype(DataT, cElT)
  return BlockSparse{cElT,cDataT,N}
end

## End set fields

# TODO: Implement as `fieldtype(storagetype, :data)`.
datatype(::Type{<:BlockSparse{<:Any,DataT}}) where {DataT} = DataT
# TODO: Implement as `ndims(blockoffsetstype(storagetype))`.
ndims(::Type{<:BlockSparse{<:Any,<:Any,N}}) where {N} = N
# TODO: Implement as `fieldtype(storagetype, :blockoffsets)`.
blockoffsetstype(storagetype::Type{<:BlockSparse}) = BlockOffsets{ndims(storagetype)}
eltype(bs::BlockSparse) = eltype(typeof(bs))

# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
function eltype(BsType::Type{<:BlockSparse})
  try
    eltype(datatype(BsType))
  catch
    throw("eltype not specified for type $(BsType)")
  end
end

eltype(::Type{BlockSparse{T}}) where {T} = T

dense(BsType::Type{<:BlockSparse}) = Dense{eltype(BsType),datatype(BsType)}

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
