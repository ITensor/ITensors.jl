# TODO: Rename `copy_nonzeros!`, move to `sparsearrayinterface.jl`.
function Base.copy!(dest::AbstractArray, src::SparseArrayDOK)
  @assert axes(dest) == axes(src)
  map!(identity, dest, src)
  return dest
end

# TODO: Rename `copyto_nonzeros!`, move to `sparsearrayinterface.jl`.
function Base.copyto!(dest::AbstractArray, src::SparseArrayDOK)
  copy!(dest, src)
  return dest
end
