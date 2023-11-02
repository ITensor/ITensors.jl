# TODO: Make `isgpu`, `ismtl`, etc.
# For `isgpu`, will require a `NDTensorsGPUArrayCoreExt`.
iscu(A::AbstractArray) = iscu(typeof(A))
function iscu(A::Type{<:AbstractArray})
  return (unwrap_type(A) == A ? false : iscu(unwrap_type(A)))
end
