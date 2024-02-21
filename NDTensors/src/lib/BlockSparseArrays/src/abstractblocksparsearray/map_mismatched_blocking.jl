using BlockArrays: BlockArrays, BlockRange

function map_mismatched_blocking!(f, a_dest::AbstractArray, a_src::AbstractArray)
  # Create a common set of axes with a blocking that includes the
  # blocking of `a_dest` and `a_src`.
  matching_axes = BlockArrays.combine_blockaxes.(axes(a_dest), axes(a_src))
  # TODO: Also include `block_stored_indices(a_dest)`!
  for b in block_stored_indices(a_src)
    # Get the subblocks of the matching axes
    # TODO: `union` all `subblocks` of all `a_src` and `a_dest`.
    subblocks = BlockRange(
      ntuple(ndims(a_dest)) do dim
        findblocks(matching_axes[dim], axes(a_src, dim)[Tuple(b)[dim]])
      end,
    )
    for subblock in subblocks
      I = cartesianindices(matching_axes, subblock)
      I_dest = blockindexrange(a_dest, I)
      I_src = blockindexrange(a_src, I)

      # TODO: Broken, need to fix.
      # map!(f, view(a_dest, I_dest), view(a_src, I_src))

      map!(f, view(a_dest, I_dest), a_src[I_src])
    end
  end
  return a_dest
end
