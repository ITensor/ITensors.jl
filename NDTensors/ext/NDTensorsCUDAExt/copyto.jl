# Catches a bug in `copyto!` in CuArray backend.
function Base.copyto!(
  dest::Exposed{<:CuArray}, src::Exposed{<:CuArray,<:Base.ReshapedArray}
)
  copyto!(dest, expose(parent(src)))
  return unexpose(dest)
end
