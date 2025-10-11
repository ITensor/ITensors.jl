using SparseArrays: nnz
using .Vendored.TypeParameterAccessors: similartype

#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT, N, StoreT, IndsT} =
    Tensor{ElT, N, StoreT, IndsT} where {StoreT <: BlockSparse}

nonzeros(T::Tensor) = data(T)

function BlockSparseTensor(
        ::Type{ElT}, ::UndefInitializer, boffs::BlockOffsets, inds
    ) where {ElT <: Number}
    nnz_tot = nnz(boffs, inds)
    storage = BlockSparse(ElT, undef, boffs, nnz_tot)
    return tensor(storage, inds)
end

function BlockSparseTensor(
        datatype::Type{<:AbstractArray}, ::UndefInitializer, boffs::BlockOffsets, inds
    )
    nnz_tot = nnz(boffs, inds)
    storage = BlockSparse(datatype, undef, boffs, nnz_tot)
    return tensor(storage, inds)
end

function BlockSparseTensor(
        ::Type{ElT}, ::UndefInitializer, blocks::Vector{BlockT}, inds
    ) where {ElT <: Number, BlockT <: Union{Block, NTuple}}
    boffs, nnz = blockoffsets(blocks, inds)
    storage = BlockSparse(ElT, undef, boffs, nnz)
    return tensor(storage, inds)
end

function BlockSparseTensor(
        datatype::Type{<:AbstractArray},
        ::UndefInitializer,
        blocks::Vector{<:Union{Block, NTuple}},
        inds,
    )
    boffs, nnz = blockoffsets(blocks, inds)
    storage = BlockSparse(datatype, undef, boffs, nnz)
    return tensor(storage, inds)
end

"""
    BlockSparseTensor(::UndefInitializer, blocks, inds)

Construct a block sparse tensor with uninitialized memory
from indices and locations of non-zero blocks.
"""
function BlockSparseTensor(::UndefInitializer, blockoffsets, inds)
    return BlockSparseTensor(default_eltype(), undef, blockoffsets, inds)
end

function BlockSparseTensor(
        datatype::Type{<:AbstractArray}, blockoffsets::BlockOffsets, inds
    )
    nnz_tot = nnz(blockoffsets, inds)
    storage = BlockSparse(datatype, blockoffsets, nnz_tot)
    return tensor(storage, inds)
end

function BlockSparseTensor(eltype::Type{<:Number}, blockoffsets::BlockOffsets, inds)
    return BlockSparseTensor(Vector{eltype}, blockoffsets, inds)
end

function BlockSparseTensor(blockoffsets::BlockOffsets, inds)
    return BlockSparseTensor(default_eltype(), blockoffsets, inds)
end

"""
    BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
BlockSparseTensor(inds) = BlockSparseTensor(default_eltype(), inds)

function BlockSparseTensor(datatype::Type{<:AbstractArray}, inds)
    return BlockSparseTensor(datatype, BlockOffsets{length(inds)}(), inds)
end

function BlockSparseTensor(eltype::Type{<:Number}, inds)
    return BlockSparseTensor(Vector{eltype}, inds)
end

"""
    BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds::Vararg{DimT, N}) where {DimT, N}
    return BlockSparseTensor(BlockOffsets{N}(), inds)
end

"""
    BlockSparseTensor(blocks::Vector{Block{N}}, inds)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Vector{BlockT}, inds) where {BlockT <: Union{Block, NTuple}}
    return BlockSparseTensor(default_eltype(), blocks, inds)
end

function BlockSparseTensor(
        ::Type{ElT}, blocks::Vector{BlockT}, inds
    ) where {ElT <: Number, BlockT <: Union{Block, NTuple}}
    boffs, nnz = blockoffsets(blocks, inds)
    storage = BlockSparse(ElT, boffs, nnz)
    return tensor(storage, inds)
end

function BlockSparseTensor(
        datatype::Type{<:AbstractArray}, blocks::Vector{<:Union{Block, NTuple}}, inds
    )
    boffs, nnz = blockoffsets(blocks, inds)
    storage = BlockSparse(datatype, boffs, nnz)
    return tensor(storage, inds)
end

function BlockSparseTensor(
        x::Number, blocks::Vector{BlockT}, inds
    ) where {BlockT <: Union{Block, NTuple}}
    boffs, nnz = blockoffsets(blocks, inds)
    storage = BlockSparse(x, boffs, nnz)
    return tensor(storage, inds)
end

#complex(::Type{BlockSparseTensor{ElT,N,StoreT,IndsT}}) where {ElT<:Number,N,StoreT<:BlockSparse
#  = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

function randn(
        TensorT::Type{<:BlockSparseTensor{ElT, N}}, blocks::Vector{<:BlockT}, inds
    ) where {ElT, BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    return randn(Random.default_rng(), TensorT, blocks, inds)
end

function randn(
        rng::AbstractRNG, ::Type{<:BlockSparseTensor{ElT, N}}, blocks::Vector{<:BlockT}, inds
    ) where {ElT, BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    boffs, nnz = blockoffsets(blocks, inds)
    storage = randn(rng, BlockSparse{ElT}, boffs, nnz)
    return tensor(storage, inds)
end

function randomBlockSparseTensor(
        ::Type{ElT}, blocks::Vector{<:BlockT}, inds
    ) where {ElT, BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    return randomBlockSparseTensor(Random.default_rng(), ElT, blocks, inds)
end

function randomBlockSparseTensor(
        rng::AbstractRNG, ::Type{ElT}, blocks::Vector{<:BlockT}, inds
    ) where {ElT, BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    return randn(rng, BlockSparseTensor{ElT, N}, blocks, inds)
end

function randomBlockSparseTensor(blocks::Vector, inds)
    return randomBlockSparseTensor(Random.default_rng(), blocks, inds)
end

function randomBlockSparseTensor(rng::AbstractRNG, blocks::Vector, inds)
    return randomBlockSparseTensor(rng, default_eltype(), blocks, inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds::BlockDims...)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(
        blocks::Vector{BlockT}, inds::Vararg{BlockDim, N}
    ) where {BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    return BlockSparseTensor(blocks, inds)
end

function BlockSparseTensor{ElT}(
        blocks::Vector{BlockT}, inds::Vararg{BlockDim, N}
    ) where {ElT <: Number, BlockT <: Union{Block{N}, NTuple{N, <:Integer}}} where {N}
    return BlockSparseTensor(ElT, blocks, inds)
end

function zeros(
        tensor::BlockSparseTensor{ElT, N}, blockoffsets::BlockOffsets{N}, inds
    ) where {ElT, N}
    return BlockSparseTensor(datatype(tensor), blockoffsets, inds)
end

function zeros(
        tensortype::Type{<:BlockSparseTensor{ElT, N}}, blockoffsets::BlockOffsets{N}, inds
    ) where {ElT, N}
    return BlockSparseTensor(datatype(tensortype), blockoffsets, inds)
end

function zeros(tensortype::Type{<:BlockSparseTensor}, inds)
    return BlockSparseTensor(datatype(tensortype), inds)
end

zeros(tensor::BlockSparseTensor, inds) = zeros(typeof(tensor), inds)

# Basic functionality for AbstractArray interface
IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the CartesianIndices for the range of indices
# of the specified
function blockindices(T::BlockSparseTensor{ElT, N}, block) where {ElT, N}
    return CartesianIndex(blockstart(T, block)):CartesianIndex(blockend(T, block))
end

"""
indexoffset(T::BlockSparseTensor,i::Int...) -> offset,block,blockoffset

Get the offset in the data of the specified
CartesianIndex. If it falls in a block that doesn't
exist, return nothing for the offset.
Also returns the block the index is found in and the offset
within the block.
"""
function indexoffset(T::BlockSparseTensor{ElT, N}, i::Vararg{Int, N}) where {ElT, N}
    index_within_block, block = blockindex(T, i...)
    block_dims = blockdims(T, block)
    offset_within_block = LinearIndices(block_dims)[CartesianIndex(index_within_block)]
    offset_of_block = offset(T, block)
    offset_of_i = isnothing(offset_of_block) ? nothing : offset_of_block + offset_within_block
    return offset_of_i, block, offset_within_block
end

# TODO: Add a checkbounds
# TODO: write this nicer in terms of blockview?
#       Could write:
#       block,index_within_block = blockindex(T,i...)
#       return blockview(T,block)[index_within_block]
@propagate_inbounds function getindex(
        T::BlockSparseTensor{ElT, N}, i::Vararg{Int, N}
    ) where {ElT, N}
    offset, _ = indexoffset(T, i...)
    isnothing(offset) && return zero(ElT)
    return storage(T)[offset]
end

@propagate_inbounds function getindex(T::BlockSparseTensor{ElT, 0}) where {ElT}
    nnzblocks(T) == 0 && return zero(ElT)
    return expose(storage(T))[]
end

# These may not be valid if the Tensor has no blocks
#@propagate_inbounds getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = storage(T)[ind]

#@propagate_inbounds getindex(T::BlockSparseTensor{<:Number,0}) = storage(T)[1]

# Add the specified block to the BlockSparseTensor
# Insert it such that the blocks remain ordered.
# Defaults to adding zeros.
# Returns the offset of the new block added.
# XXX rename to insertblock!, no need to return offset
using .Vendored.TypeParameterAccessors: unwrap_array_type
using .Expose: Exposed, expose, unexpose
function insertblock_offset!(T::BlockSparseTensor{ElT, N}, newblock::Block{N}) where {ElT, N}
    newdim = blockdim(T, newblock)
    newoffset = nnz(T)
    insert!(blockoffsets(T), newblock, newoffset)
    # Insert new block into data
    new_data = generic_zeros(unwrap_array_type(T), newdim)
    # TODO: `append!` is broken on `Metal` since `resize!`
    # isn't implemented.
    append!(expose(data(T)), new_data)
    return newoffset
end

function insertblock!(T::BlockSparseTensor{<:Number, N}, block::Block{N}) where {N}
    insertblock_offset!(T, block)
    return T
end

insertblock!(T::BlockSparseTensor, block) = insertblock!(T, Block(block))

# Insert missing diagonal blocks as zero blocks
function insert_diag_blocks!(T::AbstractArray)
    for b in eachdiagblock(T)
        blockT = blockview(T, b)
        if isnothing(blockT)
            # Block was not found in the list, insert it
            insertblock!(T, b)
        end
    end
    return nothing
end

# TODO: Add a checkbounds
@propagate_inbounds function setindex!(
        T::BlockSparseTensor{ElT, N}, val, i::Vararg{Int, N}
    ) where {ElT, N}
    offset, block, offset_within_block = indexoffset(T, i...)
    if isnothing(offset)
        offset_of_block = insertblock_offset!(T, block)
        offset = offset_of_block + offset_within_block
    end
    storage(T)[offset] = val
    return T
end

hasblock(T::Tensor, block::Block) = isassigned(blockoffsets(T), block)

@propagate_inbounds function setindex!(
        T::BlockSparseTensor{ElT, N}, val, b::Block{N}
    ) where {ElT, N}
    if !hasblock(T, b)
        insertblock!(T, b)
    end
    Tb = T[b]
    Tb .= val
    return T
end

getindex(T::BlockSparseTensor, block::Block) = blockview(T, block)

to_indices(T::Tensor{<:Any, N}, b::Tuple{Block{N}}) where {N} = blockindices(T, b...)

function blockview(T::BlockSparseTensor, block::Block)
    return blockview(T, block, offset(T, block))
end

function blockview(T::BlockSparseTensor, block::Block, offset::Integer)
    return blockview(T, BlockOffset(block, offset))
end

# Case where the block isn't found, return nothing
function blockview(T::BlockSparseTensor, block::Block, ::Nothing)
    return nothing
end

blockview(T::BlockSparseTensor, block) = blockview(T, Block(block))

function blockview(T::BlockSparseTensor, bof::BlockOffset)
    blockT, offsetT = bof
    blockdimsT = blockdims(T, blockT)
    blockdimT = prod(blockdimsT)
    dataTslice = @view data(storage(T))[(offsetT + 1):(offsetT + blockdimT)]
    return tensor(Dense(dataTslice), blockdimsT)
end

view(T::BlockSparseTensor, b::Block) = blockview(T, b)

# convert to Dense
function dense(T::TensorT) where {TensorT <: BlockSparseTensor}
    R = zeros(dense(TensorT), inds(T))
    ## Here this failed with scalar indexing (R[blockindices] = blockview)
    ## We can fix this by using copyto the arrays
    r = array(R)
    for block in keys(blockoffsets(T))
        # TODO: make sure this assignment is efficient
        rview = @view r[blockindices(T, block)]
        copyto!(expose(rview), expose(array(blockview(T, block))))
    end
    return tensor(Dense(r), inds(T))
end

function diag(ETensor::Exposed{<:AbstractArray, <:BlockSparseTensor})
    tensor = unexpose(ETensor)
    tensordiag = NDTensors.similar(
        dense(typeof(tensor)), eltype(tensor), (diaglength(tensor),)
    )
    for j in 1:diaglength(tensor)
        @inbounds tensordiag[j] = getdiagindex(tensor, j)
    end
    return tensordiag
end

function Base.mapreduce(
        f, op, t1::BlockSparseTensor, t_tail::BlockSparseTensor...; kwargs...
    )
    # TODO: Take advantage of block sparsity here.
    return mapreduce(f, op, array(t1), array.(t_tail)...; kwargs...)
end

# This is a special case that optimizes for a single tensor
# and takes advantage of block sparsity. Once the more general
# case handles block sparsity, this can be removed.
function Base.mapreduce(f, op, t::BlockSparseTensor; kwargs...)
    elt = eltype(t)
    if !iszero(f(zero(elt)))
        return mapreduce(f, op, array(t); kwargs...)
    end
    if length(t) > nnz(t)
        # Some elements are zero, account for that
        # with the initial value.
        init_kwargs = (; init = zero(elt))
    else
        init_kwargs = (;)
    end
    return mapreduce(f, op, storage(t); kwargs..., init_kwargs...)
end

function blocksparse_isequal(x, y)
    return array(x) == array(y)
end
function Base.:(==)(x::BlockSparseTensor, y::BlockSparseTensor)
    return blocksparse_isequal(x, y)
end
function Base.:(==)(x::BlockSparseTensor, y::Tensor)
    return blocksparse_isequal(x, y)
end
function Base.:(==)(x::Tensor, y::BlockSparseTensor)
    return blocksparse_isequal(x, y)
end

## TODO currently this fails on GPU with scalar indexing
function map_diag!(
        f::Function,
        exposed_t_destination::Exposed{<:AbstractArray, <:BlockSparseTensor},
        exposed_t_source::Exposed{<:AbstractArray, <:BlockSparseTensor},
    )
    t_destination = unexpose(exposed_t_destination)
    t_source = unexpose(exposed_t_source)
    for i in 1:diaglength(t_destination)
        NDTensors.setdiagindex!(t_destination, f(NDTensors.getdiagindex(t_source, i)), i)
    end
    return t_destination
end
#
# Operations
#

# TODO: extend to case with different block structures
function +(T1::BlockSparseTensor{<:Number, N}, T2::BlockSparseTensor{<:Number, N}) where {N}
    inds(T1) ≠ inds(T2) &&
        error("Cannot add block sparse tensors with different block structure")
    R = copy(T1)
    return permutedims!!(R, T2, ntuple(identity, Val(N)), +)
end

function permutedims(T::BlockSparseTensor{<:Number, N}, perm::NTuple{N, Int}) where {N}
    blockoffsetsR, indsR = permutedims(blockoffsets(T), inds(T), perm)
    R = NDTensors.similar(T, blockoffsetsR, indsR)
    permutedims!(R, T, perm)
    return R
end

function _permute_combdims(combdims::NTuple{NC, Int}, perm::NTuple{NP, Int}) where {NC, NP}
    res = MVector{NC, Int}(undef)
    iperm = invperm(perm)
    for i in 1:NC
        res[i] = iperm[combdims[i]]
    end
    return Tuple(res)
end

#
# These are functions to help with combining and uncombining
#

# Note that combdims is expected to be contiguous and ordered
# smallest to largest
function combine_dims(blocks::Vector{Block{N}}, inds, combdims::NTuple{NC, Int}) where {N, NC}
    nblcks = nblocks(inds, combdims)
    blocks_comb = Vector{Block{N - NC + 1}}(undef, length(blocks))
    for (i, block) in enumerate(blocks)
        blocks_comb[i] = combine_dims(block, inds, combdims)
    end
    return blocks_comb
end

function combine_dims(block::Block, inds, combdims::NTuple{NC, Int}) where {NC}
    nblcks = nblocks(inds, combdims)
    slice = getindices(block, combdims)
    slice_comb = LinearIndices(nblcks)[slice...]
    block_comb = deleteat(block, combdims)
    block_comb = insertafter(block_comb, tuple(slice_comb), minimum(combdims) - 1)
    return block_comb
end

# In the dimension dim, permute the blocks
function perm_blocks(blocks::Blocks{N}, dim::Int, perm) where {N}
    blocks_perm = Blocks{N}(undef, nnzblocks(blocks))
    iperm = invperm(perm)
    for (i, block) in enumerate(blocks)
        blocks_perm[i] = setindex(block, iperm[block[dim]], dim)
    end
    return blocks_perm
end

# In the dimension dim, permute the block
function perm_block(block::Block, dim::Int, perm)
    iperm = invperm(perm)
    return setindex(block, iperm[block[dim]], dim)
end

# In the dimension dim, combine the specified blocks
function combine_blocks(blocks::Blocks, dim::Int, blockcomb::Vector{Int})
    blocks_comb = copy(blocks)
    nnz_comb = nnzblocks(blocks)
    for (i, block) in enumerate(blocks)
        dimval = block[dim]
        blocks_comb[i] = setindex(block, blockcomb[dimval], dim)
    end
    unique!(blocks_comb)
    return blocks_comb
end

function permutedims_combine_output(
        T::BlockSparseTensor{ElT, N},
        is,
        perm::NTuple{N, Int},
        combdims::NTuple{NC, Int},
        blockperm::Vector{Int},
        blockcomb::Vector{Int},
    ) where {ElT, N, NC}
    # Permute the indices
    indsT = inds(T)
    inds_perm = permute(indsT, perm)

    # Now that the indices are permuted, compute
    # which indices are now combined
    combdims_perm = TupleTools.sort(_permute_combdims(combdims, perm))

    # Permute the nonzero blocks (dimension-wise)
    blocks = nzblocks(T)
    blocks_perm = permutedims(blocks, perm)

    # Combine the nonzero blocks (dimension-wise)
    blocks_perm_comb = combine_dims(blocks_perm, inds_perm, combdims_perm)

    # Permute the blocks (within the newly combined dimension)
    comb_ind_loc = minimum(combdims_perm)
    blocks_perm_comb = perm_blocks(blocks_perm_comb, comb_ind_loc, blockperm)
    blocks_perm_comb = sort(blocks_perm_comb; lt = isblockless)

    # Combine the blocks (within the newly combined and permuted dimension)
    blocks_perm_comb = combine_blocks(blocks_perm_comb, comb_ind_loc, blockcomb)

    return BlockSparseTensor(unwrap_array_type(T), blocks_perm_comb, is)
end

function permutedims_combine(
        T::BlockSparseTensor{ElT, N},
        is,
        perm::NTuple{N, Int},
        combdims::NTuple{NC, Int},
        blockperm::Vector{Int},
        blockcomb::Vector{Int},
    ) where {ElT, N, NC}
    R = permutedims_combine_output(T, is, perm, combdims, blockperm, blockcomb)

    # Permute the indices
    inds_perm = permute(inds(T), perm)

    # Now that the indices are permuted, compute
    # which indices are now combined
    combdims_perm = TupleTools.sort(_permute_combdims(combdims, perm))
    comb_ind_loc = minimum(combdims_perm)

    # Determine the new index before combining
    inds_to_combine = getindices(inds_perm, combdims_perm)
    ind_comb = ⊗(inds_to_combine...)
    ind_comb = permuteblocks(ind_comb, blockperm)

    for bof in pairs(blockoffsets(T))
        Tb = blockview(T, bof)
        b = nzblock(bof)
        b_perm = permute(b, perm)
        b_perm_comb = combine_dims(b_perm, inds_perm, combdims_perm)
        b_perm_comb = perm_block(b_perm_comb, comb_ind_loc, blockperm)
        b_in_combined_dim = b_perm_comb[comb_ind_loc]
        new_b_in_combined_dim = blockcomb[b_in_combined_dim]
        offset = 0
        pos_in_new_combined_block = 1
        while b_in_combined_dim - pos_in_new_combined_block > 0 &&
                blockcomb[b_in_combined_dim - pos_in_new_combined_block] == new_b_in_combined_dim
            offset += blockdim(ind_comb, b_in_combined_dim - pos_in_new_combined_block)
            pos_in_new_combined_block += 1
        end
        b_new = setindex(b_perm_comb, new_b_in_combined_dim, comb_ind_loc)

        Rb_total = blockview(R, b_new)
        dimsRb_tot = dims(Rb_total)
        subind = ntuple(
            i -> if i == comb_ind_loc
                range(1 + offset; stop = offset + blockdim(ind_comb, b_in_combined_dim))
            else
                range(1; stop = dimsRb_tot[i])
            end,
            N - NC + 1,
        )
        Rb = @view array(Rb_total)[subind...]

        # XXX Are these equivalent?
        #Tb_perm = permutedims(Tb,perm)
        #copyto!(Rb,Tb_perm)

        # XXX Not sure what this was for
        Rb = reshape(Rb, permute(dims(Tb), perm))
        # TODO: Make this `convert` call more general
        # for GPUs.
        Tbₐ = convert(Array, Tb)
        ## @strided Rb .= permutedims(Tbₐ, perm)
        permutedims!(expose(Rb), expose(Tbₐ), perm)
    end

    return R
end

# TODO: optimize by avoiding findfirst
function _number_uncombined(blockval::Integer, blockcomb::Vector)
    if blockval == blockcomb[end]
        return length(blockcomb) - findfirst(==(blockval), blockcomb) + 1
    end
    return findfirst(==(blockval + 1), blockcomb) - findfirst(==(blockval), blockcomb)
end

# TODO: optimize by avoiding findfirst
function _number_uncombined_shift(blockval::Integer, blockcomb::Vector)
    if blockval == 1
        return 0
    end
    ncomb_shift = 0
    for i in 1:(blockval - 1)
        ncomb_shift += findfirst(==(i + 1), blockcomb) - findfirst(==(i), blockcomb) - 1
    end
    return ncomb_shift
end

# Uncombine the blocks along the dimension dim
# according to the pattern in blockcomb (for example, blockcomb
# is [1,2,2,3] and dim = 2, so the blocks (1,2),(2,3) get
# split into (1,2),(1,3),(2,4))
function uncombine_blocks(blocks::Blocks{N}, dim::Int, blockcomb::Vector{Int}) where {N}
    blocks_uncomb = Blocks{N}()
    ncomb_tot = 0
    for i in 1:length(blocks)
        block = blocks[i]
        blockval = block[dim]
        ncomb = _number_uncombined(blockval, blockcomb)
        ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
        push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
        for j in 1:(ncomb - 1)
            push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
        end
    end
    return blocks_uncomb
end

function uncombine_block(block::Block{N}, dim::Int, blockcomb::Vector{Int}) where {N}
    blocks_uncomb = Blocks{N}()
    ncomb_tot = 0
    blockval = block[dim]
    ncomb = _number_uncombined(blockval, blockcomb)
    ncomb_shift = _number_uncombined_shift(blockval, blockcomb)
    push!(blocks_uncomb, setindex(block, blockval + ncomb_shift, dim))
    for j in 1:(ncomb - 1)
        push!(blocks_uncomb, setindex(block, blockval + ncomb_shift + j, dim))
    end
    return blocks_uncomb
end

function uncombine_output(
        T::BlockSparseTensor{ElT, N},
        T_labels,
        is,
        is_labels,
        combdim::Int,
        blockperm::Vector{Int},
        blockcomb::Vector{Int},
    ) where {ElT <: Number, N}
    labels_uncomb_perm = setdiff(is_labels, T_labels)
    ind_uncomb_perm = ⊗(is[map(x -> findfirst(==(x), is_labels), labels_uncomb_perm)]...)
    inds_uncomb_perm = insertat(inds(T), ind_uncomb_perm, combdim)
    # Uncombine the blocks of T
    blocks_uncomb = uncombine_blocks(nzblocks(T), combdim, blockcomb)
    blocks_uncomb_perm = perm_blocks(blocks_uncomb, combdim, invperm(blockperm))
    boffs_uncomb_perm, nnz_uncomb_perm = blockoffsets(blocks_uncomb_perm, inds_uncomb_perm)
    T_uncomb_perm = tensor(
        BlockSparse(unwrap_array_type(T), boffs_uncomb_perm, nnz_uncomb_perm), inds_uncomb_perm
    )
    R = reshape(T_uncomb_perm, is)
    return R
end

function reshape(blockT::Block{NT}, indsT, indsR) where {NT}
    nblocksT = nblocks(indsT)
    nblocksR = nblocks(indsR)
    blockR = Tuple(
        CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]]
    )
    return blockR
end

function uncombine(
        T::BlockSparseTensor{<:Number, NT},
        T_labels,
        is,
        is_labels,
        combdim::Int,
        blockperm::Vector{Int},
        blockcomb::Vector{Int},
    ) where {NT}
    NR = length(is)
    R = uncombine_output(T, T_labels, is, is_labels, combdim, blockperm, blockcomb)
    invblockperm = invperm(blockperm)
    # This is needed for reshaping the block
    # TODO: It is already calculated in uncombine_output, use it from there
    labels_uncomb_perm = setdiff(is_labels, T_labels)
    ind_uncomb_perm = ⊗(is[map(x -> findfirst(==(x), is_labels), labels_uncomb_perm)]...)
    ind_uncomb = permuteblocks(ind_uncomb_perm, blockperm)
    # Same as inds(T) but with the blocks uncombined
    inds_uncomb = insertat(inds(T), ind_uncomb, combdim)
    inds_uncomb_perm = insertat(inds(T), ind_uncomb_perm, combdim)
    for bof in pairs(blockoffsets(T))
        b = nzblock(bof)
        Tb_tot = blockview(T, bof)
        dimsTb_tot = dims(Tb_tot)
        bs_uncomb = uncombine_block(b, combdim, blockcomb)
        offset = 0
        for i in 1:length(bs_uncomb)
            b_uncomb = bs_uncomb[i]
            b_uncomb_perm = perm_block(b_uncomb, combdim, invblockperm)
            b_uncomb_perm_reshape = reshape(b_uncomb_perm, inds_uncomb_perm, is)
            Rb = blockview(R, b_uncomb_perm_reshape)
            b_uncomb_in_combined_dim = b_uncomb_perm[combdim]
            start = offset + 1
            stop = offset + blockdim(ind_uncomb_perm, b_uncomb_in_combined_dim)
            subind = ntuple(
                i -> i == combdim ? range(start; stop = stop) : range(1; stop = dimsTb_tot[i]), NT
            )
            offset = stop
            Tb = @view array(Tb_tot)[subind...]

            # Alternative (but maybe slower):
            #copyto!(Rb,Tb)

            if length(Tb) == 1
                # Call `cpu` to avoid allowscalar error on GPU.
                # TODO: Replace with `@allowscalar`, requires adding
                # `GPUArraysCore.jl` as a dependency.
                Rb[] = cpu(Tb)[]
            else
                # XXX: this used to be:
                # Rbₐᵣ = ReshapedArray(parent(Rbₐ), size(Tb), ())
                # however that doesn't work with subarrays
                Rbₐ = convert(Array, Rb)
                ## Rbₐᵣ = ReshapedArray(Rbₐ, size(Tb), ())
                Rbₐᵣ = reshape(Rbₐ, size(Tb))
                ## @strided Rbₐᵣ .= Tb
                copyto!(expose(Rbₐᵣ), expose(Tb))
            end
        end
    end
    return R
end

function copyto!(R::BlockSparseTensor, T::BlockSparseTensor)
    for bof in pairs(blockoffsets(T))
        copyto!(blockview(R, nzblock(bof)), blockview(T, bof))
    end
    return R
end

# TODO: handle case where:
# f(zero(ElR),zero(ElT)) != promote_type(ElR,ElT)
function permutedims!!(
        R::BlockSparseTensor{ElR, N},
        T::BlockSparseTensor{ElT, N},
        perm::NTuple{N, Int},
        f::Function = (r, t) -> t,
    ) where {ElR, ElT, N}
    RR = convert(promote_type(typeof(R), typeof(T)), R)
    permutedims!(RR, T, perm, f)
    return RR
end

# <fermions>
scale_blocks!(T, compute_fac::Function = (b) -> 1) = T

# <fermions>
function scale_blocks!(
        T::BlockSparseTensor{<:Number, N}, compute_fac::Function = (b) -> 1
    ) where {N}
    for blockT in keys(blockoffsets(T))
        fac = compute_fac(blockT)
        if fac != 1
            Tblock = blockview(T, blockT)
            scale!(Tblock, fac)
        end
    end
    return T
end

# <fermions>
permfactor(perm, block, inds) = 1

using .Vendored.TypeParameterAccessors: set_type_parameters, parenttype
function permutedims!(
        R::BlockSparseTensor{<:Number, N},
        T::BlockSparseTensor{<:Number, N},
        perm::NTuple{N, Int},
        f::Function = (r, t) -> t,
    ) where {N}
    blocks_R = keys(blockoffsets(R))
    perm_blocks_T = map(b -> permute(b, perm), keys(blockoffsets(T)))
    blocks = union(blocks_R, perm_blocks_T)
    for block in blocks
        block_T = permute(block, invperm(perm))

        # Loop over non-zero blocks of T/R
        Rblock = blockview(R, block)
        Tblock = blockview(T, block_T)

        # <fermions>
        pfac = permfactor(perm, block_T, inds(T))
        f_fac = isone(pfac) ? f : ((r, t) -> f(r, pfac * t))

        Rblock_exists = !isnothing(Rblock)
        Tblock_exists = !isnothing(Tblock)
        if !Rblock_exists
            # Rblock doesn't exist
            block_size = permute(size(Tblock), perm)
            # TODO: Make GPU friendly.
            DenseT = set_type_parameters(Dense, (eltype, parenttype), (eltype(R), datatype(R)))
            Rblock = tensor(generic_zeros(DenseT, prod(block_size)), block_size)
        elseif !Tblock_exists
            # Tblock doesn't exist
            block_size = permute(size(Rblock), invperm(perm))
            # TODO: Make GPU friendly.
            DenseT = set_type_parameters(Dense, (eltype, parenttype), (eltype(T), datatype(T)))
            Tblock = tensor(generic_zeros(DenseT, prod(block_size)), block_size)
        end
        permutedims!(Rblock, Tblock, perm, f_fac)
        if !Rblock_exists
            # Set missing nonzero block
            ## To make sure no allowscalar issue grab the data
            if !iszero(data(Rblock))
                R[block] = Rblock
            end
        end
    end
    return R
end

const IntTuple = NTuple{N, Int} where {N}
const IntOrIntTuple = Union{Int, IntTuple}

function permute_combine(inds::IndsT, pos::Vararg{IntOrIntTuple, N}) where {IndsT, N}
    IndT = eltype(IndsT)
    # Using SizedVector since setindex! doesn't
    # work for MVector when eltype not isbitstype
    newinds = SizedVector{N, IndT}(undef)
    for i in 1:N
        pos_i = pos[i]
        newind_i = inds[pos_i[1]]
        for p in 2:length(pos_i)
            newind_i = newind_i ⊗ inds[pos_i[p]]
        end
        newinds[i] = newind_i
    end
    IndsR = similartype(IndsT, Val{N})
    indsR = IndsR(Tuple(newinds))
    return indsR
end

"""
Indices are combined according to the grouping of the input,
for example (1,2),3 will combine the first two indices.
"""
function combine(inds::IndsT, com::Vararg{IntOrIntTuple, N}) where {IndsT, N}
    IndT = eltype(IndsT)
    # Using SizedVector since setindex! doesn't
    # work for MVector when eltype not isbitstype
    newinds = SizedVector{N, IndT}(undef)
    i_orig = 1
    for i in 1:N
        newind_i = inds[i_orig]
        i_orig += 1
        for p in 2:length(com[i])
            newind_i = newind_i ⊗ inds[i_orig]
            i_orig += 1
        end
        newinds[i] = newind_i
    end
    IndsR = similartype(IndsT, Val{N})
    indsR = IndsR(Tuple(newinds))
    return indsR
end

function permute_combine(
        boffs::BlockOffsets, inds::IndsT, pos::Vararg{IntOrIntTuple, N}
    ) where {IndsT, N}
    perm = flatten(pos...)
    boffsp, indsp = permutedims(boffs, inds, perm)
    indsR = combine(indsp, pos...)
    boffsR = reshape(boffsp, indsp, indsR)
    return boffsR, indsR
end

function reshape(boffsT::BlockOffsets{NT}, indsT, indsR) where {NT}
    NR = length(indsR)
    boffsR = BlockOffsets{NR}()
    nblocksT = nblocks(indsT)
    nblocksR = nblocks(indsR)
    for (blockT, offsetT) in pairs(boffsT)
        blockR = Block(
            CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]]
        )
        insert!(boffsR, blockR, offsetT)
    end
    return boffsR
end

function reshape(boffsT::BlockOffsets{NT}, blocksR::Vector{Block{NR}}) where {NR, NT}
    boffsR = BlockOffsets{NR}()
    # TODO: check blocksR is ordered and are properly reshaped
    # versions of the blocks of boffsT
    for (i, (blockT, offsetT)) in enumerate(boffsT)
        blockR = blocksR[i]
        boffsR[blockR] = offsetT
    end
    return boffsR
end

reshape(T::BlockSparse, boffsR::BlockOffsets) = BlockSparse(data(T), boffsR)

function reshape(T::BlockSparseTensor, boffsR::BlockOffsets, indsR)
    storeR = reshape(storage(T), boffsR)
    return tensor(storeR, indsR)
end

function reshape(T::BlockSparseTensor, indsR)
    # TODO: add some checks that the block dimensions
    # are consistent (e.g. nnzblocks(T) == nnzblocks(R), etc.)
    boffsR = reshape(blockoffsets(T), inds(T), indsR)
    R = reshape(T, boffsR, indsR)
    return R
end

function permute_combine(
        T::BlockSparseTensor{ElT, NT, IndsT}, pos::Vararg{IntOrIntTuple, NR}
    ) where {ElT, NT, IndsT, NR}
    boffsR, indsR = permute_combine(blockoffsets(T), inds(T), pos...)

    perm = flatten(pos...)

    length(perm) ≠ NT && error("Index positions must add up to order of Tensor ($NT)")
    isperm(perm) || error("Index positions must be a permutation")

    if !is_trivial_permutation(perm)
        Tp = permutedims(T, perm)
    else
        Tp = copy(T)
    end
    NR == NT && return Tp
    R = reshape(Tp, boffsR, indsR)
    return R
end

#
# Print block sparse tensors
#

#function summary(io::IO,
#                 T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println(io,"Number of nonzero blocks: ",nnzblocks(T))
#end

#function summary(io::IO,
#                 T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,typeof(T))
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println("Number of nonzero blocks: ",nnzblocks(T))
#end

function _range2string(rangestart::NTuple{N, Int}, rangeend::NTuple{N, Int}) where {N}
    s = ""
    for n in 1:N
        s = string(s, rangestart[n], ":", rangeend[n])
        if n < N
            s = string(s, ", ")
        end
    end
    return s
end

function Base.show(io::IO, mime::MIME"text/plain", T::BlockSparseTensor)
    summary(io, T)
    for (n, block) in enumerate(keys(blockoffsets(T)))
        blockdimsT = blockdims(T, block)
        println(io, block)
        println(io, " [", _range2string(blockstart(T, block), blockend(T, block)), "]")
        print_tensor(io, blockview(T, block))
        n < nnzblocks(T) && print(io, "\n\n")
    end
    return nothing
end

Base.show(io::IO, T::BlockSparseTensor) = show(io, MIME("text/plain"), T)
