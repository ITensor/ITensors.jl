export BlockSparseTensor,
       blockview,
       eachblock

#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

blockoffsets(T::BlockSparseTensor) = blockoffsets(store(T))
nnzblocks(T::BlockSparseTensor) = nnzblocks(store(T))
nnz(T::BlockSparseTensor) = nnz(store(T))

nblocks(T::BlockSparseTensor) = nblocks(inds(T))
blockdims(T::BlockSparseTensor{ElT,N},
          block::Block{N}) where {ElT,N} = blockdims(inds(T),block)
blockdim(T::BlockSparseTensor{ElT,N},
         block::Block{N}) where {ElT,N} = blockdim(inds(T),block)

"""
offset(T::BlockSparseTensor,
       block::Block)

Get the linear offset in the data storage for the specified block.
If the specified block is not non-zero structurally, return nothing.
"""
function offset(T::BlockSparseTensor{ElT,N},
                block::Block{N}) where {ElT,N}
  return offset(blockoffsets(T),block)
end

"""
offset(T::BlockSparseTensor,pos::Int)

Get the offset of the block at position pos
in the block-offsets list.
"""
offset(T::BlockSparseTensor,n::Int) = offset(store(T),n)

"""
blockdim(T::BlockSparseTensor,pos::Int)

Get the block dimension of the block at position pos
in the block-offset list.
"""
blockdim(T::BlockSparseTensor,pos::Int) = blockdim(store(T),pos)

findblock(T::BlockSparseTensor{ElT,N},
          block::Block{N}) where {ElT,N} = findblock(store(T),block)

new_block_pos(T::BlockSparseTensor{ElT,N},
              block::Block{N}) where {ElT,N} = new_block_pos(blockoffsets(T),block)

"""
isblocknz(T::BlockSparseTensor,
          block::Block)

Check if the specified block is non-zero
"""
isblocknz(T::BlockSparseTensor{ElT,N},
          block::Block{N}) where {ElT,N} = isblocknz(store(T),block)

function BlockSparseTensor(::Type{ElT},
                           ::UndefInitializer,
                           blockoffsets::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(blockoffsets,inds)
  storage = BlockSparse(ElT,undef,blockoffsets,nnz_tot)
  return Tensor(storage,inds)
end

function BlockSparseTensor(::UndefInitializer,
                           blockoffsets::BlockOffsets{N},
                           inds) where {N}
  return BlockSparseTensor(Float64,undef,blockoffsets,inds)
end

function BlockSparseTensor(::Type{ElT},
                           blockoffsets::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(blockoffsets,inds)
  storage = BlockSparse(ElT,blockoffsets,nnz_tot)
  return Tensor(storage,inds)
end

function BlockSparseTensor(blockoffsets::BlockOffsets{N},
                           inds) where {N}
  return BlockSparseTensor(Float64,blockoffsets,inds)
end

"""
BlockSparseTensor(::UndefInitializer,
                  blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with uninitialized memory
from indices and locations of non-zero blocks.
"""
function BlockSparseTensor(::UndefInitializer,
                           blocks::Vector{Block{N}},
                           inds) where {N}
  blockoffsets,nnz = get_blockoffsets(blocks,inds)
  storage = BlockSparse(undef,blockoffsets,nnz)
  return Tensor(storage,inds)
end

function BlockSparseTensor(::UndefInitializer,
                           blocks::Vector{Block{N}},
                           inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(undef,blocks,inds)
end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds)
  return BlockSparseTensor(BlockOffsets{length(inds)}(),inds)
end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(BlockOffsets{N}(),inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{Block{N}},
                           inds) where {N}
  blockoffsets,offset_total = get_blockoffsets(blocks,inds)
  storage = BlockSparse(blockoffsets,offset_total)
  return Tensor(storage,inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds...)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{Block{N}},
                           inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(blocks,inds)
end

function Base.similar(::BlockSparseTensor{ElT,N},
                      blockoffsets::BlockOffsets{N},
                      inds) where {ElT,N}
  return BlockSparseTensor(ElT,undef,blockoffsets,inds)
end

function Base.similar(::Type{<:BlockSparseTensor{ElT,N}},
                      blockoffsets::BlockOffsets{N},
                      inds) where {ElT,N}
  return BlockSparseTensor(ElT,undef,blockoffsets,inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Given a CartesianIndex in the range dims(T), get the block it is in
# and the index within that block
function blockindex(T::BlockSparseTensor{ElT,N},
                    i::Vararg{Int,N}) where {ElT,N}
  # Start in the (1,1,...,1) block
  current_block_loc = @MVector ones(Int,N)
  current_block_dims = blockdims(T,Tuple(current_block_loc))
  block_index = MVector(i)
  for dim in 1:N
    while block_index[dim] > current_block_dims[dim]
      block_index[dim] -= current_block_dims[dim]
      current_block_loc[dim] += 1
      current_block_dims = blockdims(T,Tuple(current_block_loc))
    end
  end
  return Block{N}(block_index),Tuple(current_block_loc)
end

# Get the starting index of the block
function blockstart(T::BlockSparseTensor{ElT,N},
                    block::Block{N}) where {ElT,N}
  start_index = @MVector ones(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]-1
      start_index[j] += blockdim(ind_j,block_j)
    end
  end
  return CartesianIndex(Tuple(start_index))
end

# Get the ending index of the block
function blockend(T::BlockSparseTensor{ElT,N},
                  block) where {ElT,N}
  end_index = @MVector zeros(Int,N)
  for j in 1:N
    ind_j = ind(T,j)
    for block_j in 1:block[j]
      end_index[j] += blockdim(ind_j,block_j)
    end
  end
  return CartesianIndex(Tuple(end_index))
end

# Get the CartesianIndices for the range of indices
# of the specified
function blockindices(T::BlockSparseTensor{ElT,N},
                      block) where {ElT,N}
  return blockstart(T,block):blockend(T,block)
end

"""
indexoffset(T::BlockSparseTensor,i::Int...) -> offset,block,blockoffset

Get the offset in the data of the specified
CartesianIndex. If it falls in a block that doesn't
exist, return nothing for the offset.
Also returns the block the index is found in and the offset
within the block.
"""
function indexoffset(T::BlockSparseTensor{ElT,N},
                     i::Vararg{Int,N}) where {ElT,N}
  index_within_block,block = blockindex(T,i...)
  block_dims = blockdims(T,block)
  offset_within_block = LinearIndices(block_dims)[CartesianIndex(index_within_block)]
  offset_of_block = offset(T,block)
  offset_of_i = isnothing(offset_of_block) ? nothing : offset_of_block+offset_within_block
  return offset_of_i,block,offset_within_block
end

# TODO: Add a checkbounds
# TODO: write this nicer in terms of blockview?
#       Could write: 
#       block,index_within_block = blockindex(T,i...)
#       return blockview(T,block)[index_within_block]
Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  offset,_ = indexoffset(T,i...)
  isnothing(offset) && return zero(ElT)
  return store(T)[offset]
end

# These may not be valid if the Tensor has no blocks
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]

#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,0}) = store(T)[1]

# Add the specified block to the BlockSparseTensor
# Insert it such that the blocks remain ordered.
# Defaults to adding zeros.
function addblock!(T::BlockSparseTensor{ElT,N},
                   newblock::Block{N}) where {ElT,N}
  newdim = blockdim(T,newblock)
  newpos = new_block_pos(T,newblock)
  newoffset = 0
  if nnzblocks(T)>0
    newoffset = offset(T,newpos-1)+blockdim(T,newpos-1)
  end
  insert!(blockoffsets(T),newpos,BlockOffset{N}(newblock,newoffset))
  splice!(data(store(T)),newoffset+1:newoffset,zeros(ElT,newdim))
  for i in newpos+1:nnzblocks(T)
    block_i,offset_i = blockoffsets(T)[i]
    blockoffsets(T)[i] = BlockOffset{N}(block_i,offset_i+newdim)
  end
  return newoffset
end

# TODO: Add a checkbounds
Base.@propagate_inbounds function Base.setindex!(T::BlockSparseTensor{ElT,N},
                                                 val,
                                                 i::Vararg{Int,N}) where {ElT,N}
  offset,block,offset_within_block = indexoffset(T,i...)
  if isnothing(offset)
    offset_of_block = addblock!(T,block)
    offset = offset_of_block+offset_within_block
  end
  store(T)[offset] = val
  return T
end

"""
blockview(T::BlockSparseTensor,block::Block)

Given a specified block, return a Dense Tensor that is a view to the data
in that block
"""
function blockview(T::BlockSparseTensor{ElT,N},
                   block::Block{N}) where {ElT,N}
  pos = findblock(T,block)
  return blockview(T,pos)
end

"""
blockview(T::BlockSparseTensor,pos::Int)

Given a specified position in the block-offset list, return a Dense Tensor 
that is a view to the data in that block (to avoid block lookup if the position
is known already).
"""
function blockview(T::BlockSparseTensor,
                   pos::Union{Int,Nothing})
  isnothing(pos) && error("Block must be structurally non-zero to get a view")
  blockoffsetT = offset(T,pos)
  blockT = block(blockoffsets(T)[pos])
  blockdimsT = blockdims(T,blockT)
  dataTslice = @view data(store(T))[blockoffsetT+1:blockoffsetT+prod(blockdimsT)]
  return Tensor(Dense(dataTslice),blockdimsT)
end

# TODO: this is not working right now
#struct EachBlock{ElT,N,StoreT,IndsT}
#  T::BlockSparseTensor{ElT,N,StoreT,IndsT}
#end
#
#function Base.iterate(iter::EachBlock,state=(blockview(iter.T,1),1))
#  block,ind = state
#  ind > nnzblocks(iter.T) && return nothing
#  return blockview(iter.T,ind),ind+1
#end
#
#function eachblock(T::BlockSparseTensor)
#  return EachBlock(T)
#end

# convert to Dense
function dense(T::TensorT) where {TensorT<:BlockSparseTensor}
  R = zeros(dense(TensorT),dense(inds(T)))
  for (block,offset) in blockoffsets(T)
    # TODO: make sure this assignment is efficient
    R[blockindices(T,block)] = blockview(T,block)
  end
  return R
end

#
# Operations
#

# TODO: extend to case with different block structures
function Base.:+(T1::BlockSparseTensor,T2::BlockSparseTensor)
  inds(T1) ≠ inds(T2) && error("Cannot add block sparse tensors with different block structure")  
  return Tensor(store(T1)+store(T2),inds(T1))
end

function Base.permutedims(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  blockoffsetsR,indsR = permute(blockoffsets(T),inds(T),perm)
  R = similar(T,blockoffsetsR,indsR)
  permutedims!(R,T,perm)
  return R
end

# TODO: handle case with different element types in R and T
#function permutedims!!(R::BlockSparseTensor{<:Number,N},
#                       T::BlockSparseTensor{<:Number,N},
#                       perm::NTuple{N,Int}) where {N}
#  blockoffsetsTp,indsTp = permute(blockoffsets(T),inds(T),perm)
#  if blockoffsetsTp == blockoffsets(R)
#    R = permutedims!(R,T,perm)
#    return R
#  end
#  R = similar(T,blockoffsetsTp,indsTp)
#  permutedims!(R,T,perm)
#  return R
#end

# TODO: handle case with different element types in R and T
function permutedims!!(R::BlockSparseTensor{<:Number,N},
                       T::BlockSparseTensor{<:Number,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {N}
  blockoffsetsTp,indsTp = permute(blockoffsets(T),inds(T),perm)
  indsTp != inds(R) && error("In permutedims!!, output indices are not permutation of input")
  if blockoffsetsTp == blockoffsets(R)
    R = permutedims!(R,T,perm,f)
    return R
  end
  R = similar(T,blockoffsetsTp,indsTp)
  permutedims!(R,T,perm,f)
  return R
end

function Base.permutedims!(R::BlockSparseTensor{<:Number,N},
                           T::BlockSparseTensor{<:Number,N},
                           perm::NTuple{N,Int},
                           f::Function=(r,t)->t) where {N}
  for (blockT,_) in blockoffsets(T)
    # Loop over non-zero blocks of T/R
    Tblock = blockview(T,blockT)
    Rblock = blockview(R,permute(blockT,perm))
    permutedims!(Rblock,Tblock,perm,f)
  end
  return R
end

#function Base.permutedims!(R::BlockSparseTensor{<:Number,N},
#                           T::BlockSparseTensor{<:Number,N},
#                           perm::NTuple{N,Int},
#                           f::Function) where {N}
#  for (blockT,_) in blockoffsets(T)
#    # Loop over non-zero blocks of T/R
#    Tblock = blockview(T,blockT)
#    Rblock = blockview(R,permute(blockT,perm))
#    permutedims!(Rblock,Tblock,perm,f)
#  end
#  return R
#end

#
# Contraction
#

# TODO: complete this function: determine the output blocks from the input blocks
# Also, save the contraction list (which block-offsets contract with which),
# may not be generic with other contraction functions!
function contraction_output(T1::TensorT1,
                            T2::TensorT2,
                            indsR::IndsR) where {TensorT1<:BlockSparseTensor,
                                                 TensorT2<:BlockSparseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  return similar(TensorR,blockoffsetsR,indsR)
end

"""
find_matching_positions(t1,t2) -> t1_to_t2

In a tuple of length(t1), store the positions in t2
where the element of t1 is found. Otherwise, store 0
to indicate that the element of t1 is not in t2.

For example, for all t1[pos1] == t2[pos2], t1_to_t2[pos1] == pos2,
otherwise t1_to_t2[pos1] == 0.
"""
function find_matching_positions(t1,t2)
  t1_to_t2 = @MVector zeros(Int,length(t1))
  for pos1 in 1:length(t1)
    for pos2 in 1:length(t2)
      if t1[pos1] == t2[pos2]
        t1_to_t2[pos1] = pos2
      end
    end
  end
  return Tuple(t1_to_t2)
end

function contract_labels(labels1,labels2,labelsR)
  labels1_to_labels2 = find_matching_positions(labels1,labels2)
  labels1_to_labelsR = find_matching_positions(labels1,labelsR)
  labels2_to_labelsR = find_matching_positions(labels2,labelsR)
  return labels1_to_labels2,labels1_to_labelsR,labels2_to_labelsR
end

function are_blocks_contracted(block1::Block{N1},
                               block2::Block{N2},
                               labels1_to_labels2::NTuple{N1,Int}) where {N1,N2}
  for i1 in 1:N1
    i2 = labels1_to_labels2[i1]
    if i2 > 0
      # This dimension is contracted
      if block1[i1] != block2[i2]
        return false
      end
    end
  end
  return true
end

function contract_blocks(block1::Block{N1},
                         labels1_to_labelsR,
                         block2::Block{N2},
                         labels2_to_labelsR,
                         ::Val{NR}) where {N1,N2,NR}
  blockR = @MVector zeros(Int,NR)
  for i1 in 1:N1
    iR = labels1_to_labelsR[i1]
    if iR > 0
      blockR[iR] = block1[i1]
    end
  end
  for i2 in 1:N2
    iR = labels2_to_labelsR[i2]
    if iR > 0
      blockR[iR] = block2[i2]
    end
  end
  return Tuple(blockR)    
end

function contract_blockoffsets(boffs1::BlockOffsets{N1},inds1,labels1,
                               boffs2::BlockOffsets{N2},inds2,labels2,
                               indsR,labelsR) where {N1,N2}
  NR = length(labelsR)
  ValNR = ValLength(labelsR)
  labels1_to_labels2,labels1_to_labelsR,labels2_to_labelsR = contract_labels(labels1,labels2,labelsR)
  blocksR = Block{NR}[]
  contraction_plan = Tuple{Int,Int,Int}[]
  for (pos1,(block1,offset1)) in enumerate(boffs1)
    for (pos2,(block2,offset2)) in enumerate(boffs2)
      if are_blocks_contracted(block1,block2,labels1_to_labels2)
        blockR = contract_blocks(block1,labels1_to_labelsR,
                                 block2,labels2_to_labelsR,
                                 ValNR)
        push!(contraction_plan,(pos1,pos2,0))
        push!(blocksR,blockR)
      end
    end
  end

  sorted_blocksR = sort(blocksR;lt=isblockless)
  unique!(sorted_blocksR)
  blockoffsetsR = BlockOffsets{NR}(undef,length(sorted_blocksR))
  nnzR = 0
  for (i,blockR) in enumerate(sorted_blocksR)
    blockoffsetsR[i] = BlockOffset(blockR,nnzR)
    nnzR += blockdim(indsR,blockR)
  end

  # Now get the locations of the output blocks
  # in the sorted block-offsets list
  for (i,blockR) in enumerate(blocksR)
    posR = findblock(blockoffsetsR,blockR)
    pos1,pos2,_ = contraction_plan[i]
    contraction_plan[i] = (pos1,pos2,posR)
  end

  return blockoffsetsR,contraction_plan
end

function contraction_output(T1::TensorT1,
                            labelsT1,
                            T2::TensorT2,
                            labelsT2,
                            labelsR) where {TensorT1<:BlockSparseTensor,
                                            TensorT2<:BlockSparseTensor}
  indsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2,labelsR)
  TensorR = contraction_output_type(TensorT1,TensorT2,typeof(indsR))
  blockoffsetsR,contraction_plan = contract_blockoffsets(blockoffsets(T1),inds(T1),labelsT1,
                                                         blockoffsets(T2),inds(T2),labelsT2,
                                                         indsR,labelsR)
  R = similar(TensorR,blockoffsetsR,indsR)
  return R,contraction_plan
end

function contract(T1::BlockSparseTensor{<:Any,N1},
                  labelsT1,
                  T2::BlockSparseTensor{<:Any,N2},
                  labelsT2,
                  labelsR = contract_labels(labelsT1,labelsT2)) where {N1,N2}
  R,contraction_plan = contraction_output(T1,labelsT1,T2,labelsT2,labelsR)
  R = contract!(R,labelsR,T1,labelsT1,T2,labelsT2,contraction_plan)
  return R
end

function contract!(R::BlockSparseTensor{<:Number,NR},
                   labelsR,
                   T1::BlockSparseTensor{<:Number,N1},
                   labelsT1,
                   T2::BlockSparseTensor{<:Number,N2},
                   labelsT2,
                   contraction_plan) where {N1,N2,NR}
  already_written_to = fill(false,nnzblocks(R))
  # In R .= α .* (T1 * T2) .+ β .* R
  α = 1
  for (pos1,pos2,posR) in contraction_plan
    blockT1 = blockview(T1,pos1)
    blockT2 = blockview(T2,pos2)
    blockR = blockview(R,posR)
    β = 1
    if !already_written_to[posR]
      already_written_to[posR] = true
      # Overwrite the block of R
      β = 0
    end
    contract!(blockR,labelsR,
              blockT1,labelsT1,
              blockT2,labelsT2,
              α,β)
  end
  return R
end

const IntTuple = NTuple{N,Int} where N
const IntOrIntTuple = Union{Int,IntTuple}

function ⊗(dim1::BlockDim,dim2::BlockDim)
  dimR = BlockDim(undef,nblocks(dim1)*nblocks(dim2))
  for (i,t) in enumerate(Iterators.product(dim1,dim2))
    dimR[i] = prod(t)
  end
  return dimR
end

function permute_combine(inds::IndsT,
                         pos::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  for i ∈ 1:N
    pos_i = pos[i]
    newind_i = inds[pos_i[1]]
    for p in 2:length(pos_i)
      newind_i = newind_i ⊗ inds[pos_i[p]]
    end
    newinds[i] = newind_i
  end
  IndsR = similar_type(IndsT,Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

"""
Indices are combined according to the grouping of the input,
for example (1,2),3 will combine the first two indices.
"""
function combine(inds::IndsT,
                 com::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  i_orig = 1
  for i ∈ 1:N
    newind_i = inds[i_orig]
    i_orig += 1
    for p in 2:length(com[i])
      newind_i = newind_i ⊗ inds[i_orig]
      i_orig += 1
    end
    newinds[i] = newind_i
  end
  IndsR = similar_type(IndsT,Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

function permute_combine(boffs::BlockOffsets,
                         inds::IndsT,
                         pos::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  perm = tuplecat(pos...)
  boffsp,indsp = permute(boffs,inds,perm)
  indsR = combine(indsp,pos...)
  boffsR = reshape(boffsp,indsp,indsR)
  return boffsR,indsR
end

function Base.reshape(boffsT::BlockOffsets{NT},
                      indsT,
                      indsR) where {NT}
  NR = length(indsR)
  boffsR = BlockOffsets{NR}(undef,nnzblocks(boffsT))
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  for (i,(blockT,offsetT)) in enumerate(boffsT)
    blockR = Tuple(CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]])
    boffsR[i] = blockR => offsetT
  end
  return boffsR
end

function Base.reshape(boffsT::BlockOffsets{NT},
                      blocksR::Vector{Block{NR}}) where {NR,NT}
  boffsR = BlockOffsets{NR}(undef,nnzblocks(boffsT))
  # TODO: check blocksR is ordered and are properly reshaped
  # versions of the blocks of boffsT
  for (i,(blockT,offsetT)) in enumerate(boffsT)
    blockR = blocksR[i]
    boffsR[i] = BlockOffset(blockR,offsetT)
  end
  return boffsR
end

function Base.reshape(T::BlockSparse,
                      boffsR::BlockOffsets)
  return BlockSparse(data(T),boffsR)
end

function Base.reshape(T::BlockSparseTensor,
                      boffsR::BlockOffsets,
                      indsR)
  storeR = reshape(store(T),boffsR)
  return Tensor(storeR,indsR)
end

function Base.reshape(T::BlockSparseTensor,
                      indsR)
  boffsR = reshape(blockoffsets(T),inds(T),indsR)
  R = reshape(T,boffsR,indsR)
  return R
end

function permute_combine(T::BlockSparseTensor{ElT,NT,IndsT},
                         pos::Vararg{IntOrIntTuple,NR}) where {ElT,NT,IndsT,NR}
  boffsR,indsR = permute_combine(blockoffsets(T),inds(T),pos...)

  perm = tuplecat(pos...)

  length(perm)≠NT && error("Index positions must add up to order of Tensor ($NT)")
  isperm(perm) || error("Index positions must be a permutation")

  if !is_trivial_permutation(perm)
    Tp = permutedims(T,perm)
  else
    Tp = copy(T)
  end
  NR==NT && return Tp
  R = reshape(Tp,boffsR,indsR)
  return R
end

#
# Print block sparse tensors
#

function Base.summary(io::IO,
                      T::BlockSparseTensor{ElT,N}) where {ElT,N}
  println(io,Base.dims2string(dims(T))," ",typeof(T))
  for (dim,ind) in enumerate(inds(T))
    println(io,"Dim $dim: ",ind)
  end
  println("Number of nonzero blocks: ",nnzblocks(T))
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::BlockSparseTensor)
  summary(io,T)
  println(io)
  for (block,_) in blockoffsets(T)
    blockdimsT = blockdims(T,block)
    # Print the location of the current block
    println(io,"Block: ",block)
    # Print the dimension of the current block
    println(io," ",Base.dims2string(blockdimsT))
    # TODO: replace with a Tensor show method
    # instead of converting to array (for other
    # storage types, like CuArray)
    Tblock = array(blockview(T,block))
    Base.print_array(io,Tblock)
    println(io)
    println(io)
  end
end

Base.show(io::IO, T::BlockSparseTensor) = show(io,MIME("text/plain"),T)

