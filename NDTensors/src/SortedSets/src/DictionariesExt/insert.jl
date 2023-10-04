SmallVectors.insert(inds::AbstractIndices, i) = insert!(copy(inds), i)
SmallVectors.delete(inds::AbstractIndices, i) = delete!(copy(inds), i)
