function fusedims(a::AbstractArray, subperms::Tuple...)
  @assert ndims(a) == sum(length, subperms)
  perm = tuple_cat(subperms...)
  @assert isperm(perm)
  # TODO: Do this lazily?
  a_permuted = permutedims(a, perm)
  sublengths = length.(subperms)
  substops = cumsum(sublengths)
  substarts = (1, (Base.front(substops) .+ 1)...)
  subranges = range.(substarts, substops)
  @show subranges
  # Get a naive product of the axes in the subrange
  axes_prod = map(subranges) do subrange
    return ⊗(map(i -> axes(a_permuted, i), subrange)...)
  end
  a_reshaped = reshape(a_permuted, axes_prod)
  # Permute and merge the axes
  mergeperms = blockmergesortperm.(axes_prod)
  block_perms = map(mergeperm -> Block.(reduce(vcat, mergeperm)), mergeperms)
  a_blockperm = getindices(a_reshaped, block_perms...)
  axes_merged = blockmerge.(axes_prod, mergeperms)
  # TODO: Use `similar` here to preserve the type of `a`.
  a_merged = BlockSparseArray{eltype(a)}(blocklengths.(axes_merged)...)
  # TODO: Make this take advantage of the sparsity of `a_blockperm`.
  copyto!(a_merged, a_blockperm)
  return a_merged
end

function matricize(a::AbstractArray, left_dims::Tuple, right_dims::Tuple)
  return fusedims(a, left_dims, right_dims)
end

## using NDTensors.BlockSparseArrays
## using Dictionaries: Dictionary
## using NDTensors.BlockSparseArrays: nonzero_blockkeys, blocktuple
## using NDTensors: NDTensors, ⊗
## using BlockArrays: BlockArrays, Block, BlockedUnitRange, blockedrange, blocklengths, blocklasts, blockfirsts, BlockVector, blocks, blockaxes, AbstractBlockLayout
## using ArrayLayouts: ArrayLayouts, MemoryLayout
## using SplitApplyCombine: SplitApplyCombine, group, groupcount, groupreduce
## using ITensors: QN

## struct BlockSparseLayout{ArrLay,BlockLay} <: AbstractBlockLayout end
## 
## function fuse(s1::AbstractUnitRange, s2::AbstractUnitRange)
##   return Base.OneTo(length(s1) * length(s2))
## end
## 
## function fuse(s1::BlockedUnitRange, s2::BlockedUnitRange)
##   return s1 ⊗ s2
## end

## struct GradedBlockedUnitRange{T,G} <: AbstractUnitRange{Int}
##   blockedrange::BlockedUnitRange{T}
##   grades::Vector{G}
## end
## gradedblockedrange(lengths::Vector{<:Integer}, grades::Vector) = GradedBlockedUnitRange(blockedrange(lengths), grades)
## BlockArrays.blockaxes(a::GradedBlockedUnitRange) = blockaxes(a.blockedrange)
## Base.getindex(a::GradedBlockedUnitRange, b::Block{1}) = a.blockedrange[b]
## BlockArrays.blockfirsts(a::GradedBlockedUnitRange) = blockfirsts(a.blockedrange)
## BlockArrays.blocklasts(a::GradedBlockedUnitRange) = blocklasts(a.blockedrange)
## BlockArrays.findblock(a::GradedBlockedUnitRange, k) = findblock(a.blockedrange, k)
## 
## Base.getindex(a::GradedBlockedUnitRange, I::Integer) = a.blockedrange[I]
## Base.first(a::GradedBlockedUnitRange) = first(a.blockedrange)
## Base.last(a::GradedBlockedUnitRange) = last(a.blockedrange)
## Base.length(a::GradedBlockedUnitRange) = length(a.blockedrange)
## Base.step(a::GradedBlockedUnitRange) = step(a.blockedrange)
## Base.unitrange(b::GradedBlockedUnitRange) = first(b):last(b)
## 
## # TODO: Make `grades` a `Dictionary` with keys of `Block{1}`?
## grade(a::GradedBlockedUnitRange, b::Block{1}) = a.grades[Int(b)]

## # Slicing
## using BlockArrays: BlockRange, _BlockedUnitRange
## Base.@propagate_inbounds function Base.getindex(b::GradedBlockedUnitRange, KR::BlockRange{1})
##   cs = blocklasts(b)
##   isempty(KR) && return _BlockedUnitRange(1,cs[1:0])
##   K,J = first(KR),last(KR)
##   k,j = Integer(K),Integer(J)
##   bax = blockaxes(b,1)
##   @boundscheck K in bax || throw(BlockBoundsError(b,K))
##   @boundscheck J in bax || throw(BlockBoundsError(b,J))
##   K == first(bax) && return _BlockedUnitRange(first(b),cs[k:j])
##   _BlockedUnitRange(cs[k-1]+1,cs[k:j])
## end
## 
## Base.show(io::IO, mimetype::MIME"text/plain", a::GradedBlockedUnitRange) =
##   Base.show(io, mimetype, a.blockedrange)
## 
## # Fuse the blocks, sorting and merging according to the grades.
## function NDTensors.outer(s1::GradedBlockedUnitRange, s2::GradedBlockedUnitRange)
##   fused_range = s1.blockedrange ⊗ s2.blockedrange
##   fused_grades = vec(map(sum, Iterators.product(s1.grades, s2.grades)))
##   return GradedBlockedUnitRange(fused_range, fused_grades)
## end
## 
## function blockmerge(s::GradedBlockedUnitRange, grouped_perm::Vector{Vector{Int}})
##   merged_grades = map(group -> s.grades[first(group)], grouped_perm)
##   lengths = blocklengths(s)
##   merged_lengths = map(group -> sum(@view(lengths[group])), grouped_perm)
##   return gradedblockedrange(merged_lengths, merged_grades)
## end

## # Sort and merge by the grade of the blocks.
## function blockmergesort(s::GradedBlockedUnitRange)
##   grouped_perm = blockmergesortperm(s)
##   return blockmerge(s, grouped_perm)
## end
## 
## # Get the permutation for sorting, then group by common elements.
## # groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
## function blockmergesortperm(s::GradedBlockedUnitRange)
##   return groupsortperm(s.grades)
## end

## function sub_axis(a::GradedBlockedUnitRange, blocks)
##   return GradedBlockedUnitRange(sub_axis(a.blockedrange, blocks), map(b -> grade(a, b), blocks))
## end

## # TODO: Delete
## function NDTensors.outer(s1::Base.OneTo, s2::Base.OneTo)
##   return Base.OneTo(length(s1) * length(s2))
## end
## 
## function blockmerge(s::Base.OneTo, grouped_perm::Vector{Vector{Int}})
##   @assert grouped_perm == [[1]]
##   return s
## end
## 
## blockmergesortperm(s::Base.OneTo) = [[1]]
## 
## function groupsorted(v)
##   return groupcount(identity, v)
## end
## 
## # Get the permutation for sorting, then group by common elements.
## # groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
## function groupsortperm(v)
##   perm = sortperm(v)
##   v_sorted = @view v[perm]
##   group_lengths = groupsorted(v_sorted)
##   return blocks(BlockVector(perm, collect(group_lengths)))
## end
## 
## tuple_cat(ts::Tuple...) = reduce((x, y) -> (x..., y...), ts)

## function sub_axis(a::BlockedUnitRange, blocks)
##   return blockedrange([length(a[b]) for b in blocks])
## end
## 
## function sub_axes(axes_src::Tuple, axes_parent::Tuple)
##   return sub_axis.(axes_src, axes_parent)
## end

## function Base.axes(a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}})
##   axes_src = axes(parent(a))
##   axes_parent = parentindices(a)
##   return sub_axes(axes_src, axes_parent)
## end
## 
## # Map the location in the sliced array of a parent block `b` to the location
## # in a block slice. Return `nothing` if the block
## # isn't in the slice.
## function sub_blockkey(a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}}, b::Block)
##   btuple = blocktuple(b)
##   dims_a = ntuple(identity, ndims(a))
##   subblock_ranges = parentindices(a)
##   I = map(i -> findfirst(==(btuple[i]), subblock_ranges[i]), dims_a)
##   if any(isnothing, I)
##     return nothing
##   end
##   return Block(I)
## end
## 
## function nonzero_sub_blockkeys(a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}})
##   sub_blockkeys_a = Block{ndims(a),Int}[]
##   parent_blockkeys_a = Block{ndims(a),Int}[]
##   for b in nonzero_blockkeys(parent(a))
##     sub_b = sub_blockkey(a, b)
##     if !isnothing(sub_b)
##       push!(parent_blockkeys_a, b)
##       push!(sub_blockkeys_a, sub_b)
##     end
##   end
##   return Dictionary(parent_blockkeys_a, sub_blockkeys_a)
## end

## # TODO: Look at what `BlockArrays.jl` does for block slicing, i.e.
## # ArrayLayouts.sub_materialize
## # ArrayLayouts.MemoryLayout, BlockLayout, BlockSparseLayout
## # etc.
## function Base.copy(a::SubArray{<:Any,<:Any,<:BlockSparseArray,<:Tuple{Vararg{<:Vector{<:Block}}}})
##   axes_dest = axes(a)
##   nonzero_sub_blockkeys_a = nonzero_sub_blockkeys(a)
##   # TODO: Preserve the grades here.
##   a_dest = BlockSparseArray{eltype(a)}(blocklengths.(axes_dest)...)
##   for b in keys(nonzero_sub_blockkeys_a)
##     a_dest[nonzero_sub_blockkeys_a[b]] = parent(a)[b]
##   end
##   return a_dest
## end
## 
## # Permute the blocks
## function getindices(a::BlockSparseArray, b::Vector{<:Block{1}}...)
##   # Lazy version
##   ap = @view a[b...]
##   return copy(ap)
## end

## d = [2, 3]
## i = blockedrange(d)
## sectors = [QN(0), QN(1)]
## ig = gradedblockedrange(d, sectors)
## 
## B = BlockSparseArray{Float64}(ig, ig, ig, ig)
## B[Block(1, 1, 1, 1)] = randn(2, 2, 2, 2)
## B[Block(2, 2, 2, 2)] = randn(3, 3, 3, 3)
## @show axes(B)
## @show length(nonzero_blockkeys(B))
## 
## B_sub = getindices(B, [Block(2)], [Block(2)], [Block(2)], [Block(2)])
## @show B[Block(2, 2, 2, 2)] == B_sub[Block(1, 1, 1, 1)]
## @show length(nonzero_blockkeys(B_sub))
## 
## B_fused = fusedims(B, (1, 2), (3, 4))
## @show axes(B_fused)
## @show length(nonzero_blockkeys(B_fused))
