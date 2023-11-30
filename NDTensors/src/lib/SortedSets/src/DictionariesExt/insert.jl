SmallVectors.insert(inds::AbstractIndices, i) = insert(InsertStyle(inds), inds, i)

function SmallVectors.insert(::InsertStyle, inds::AbstractIndices, i)
  return error("Not implemented")
end

function SmallVectors.insert(::IsInsertable, inds::AbstractIndices, i)
  inds = copy(inds)
  insert!(inds, i)
  return inds
end

function SmallVectors.insert(::FastCopy, inds::AbstractIndices, i)
  minds = thaw(inds)
  insert!(minds, i)
  return freeze(minds)
end

SmallVectors.delete(inds::AbstractIndices, i) = delete(InsertStyle(inds), inds, i)

function SmallVectors.delete(::InsertStyle, inds::AbstractIndices, i)
  return error("Not implemented")
end

function SmallVectors.delete(::IsInsertable, inds::AbstractIndices, i)
  inds = copy(inds)
  delete!(inds, i)
  return inds
end

function SmallVectors.delete(::FastCopy, inds::AbstractIndices, i)
  minds = thaw(inds)
  delete!(minds, i)
  return freeze(minds)
end
