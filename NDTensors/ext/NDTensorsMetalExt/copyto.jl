# Catches a bug in `copyto!` in Metal backend.
function NDTensors.copyto!(
  ::Type{<:MtlArray}, dest::AbstractArray, ::Type{<:MtlArray}, src::SubArray
)
  return Base.copyto!(dest, copy(src))
end

# Catches a bug in `copyto!` in Metal backend.
function NDTensors.copyto!(
  ::Type{<:MtlArray}, dest::AbstractArray, ::Type{<:MtlArray}, src::Base.ReshapedArray
)
  return NDTensors.copyto!(dest, parent(src))
end
