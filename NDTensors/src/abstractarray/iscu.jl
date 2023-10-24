# TODO: Make `isgpu`, `ismtl`, etc.
# For `isgpu`, will require a `NDTensorsGPUArrayCoreExt`.
iscu(A::AbstractArray) = iscu(typeof(A))
function iscu(A::Type{<:AbstractArray})
  return (leaf_parenttype(A) == A ? false : iscu(leaf_parenttype(A)))
end
