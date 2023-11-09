function Base.copy(src::Exposed{<:MtlArray,<:Base.ReshapedArray})
  return reshape(copy(parent(src)), size(unexpose(src)))
end

function Base.copy(
  src::Exposed{
    <:MtlArray,<:SubArray{<:Any,<:Any,<:Base.ReshapedArray{<:Any,<:Any,<:Adjoint}}
  },
)
  return copy(@view copy(expose(parent(src)))[parentindices(unexpose(src))...])
end

# Catches a bug in `copyto!` in Metal backend.
function Base.copyto!(dest::Exposed{<:MtlArray}, src::Exposed{<:MtlArray,<:SubArray})
  return copyto!(dest, expose(copy(src)))
end

# Catches a bug in `copyto!` in Metal backend.
function Base.copyto!(
  dest::Exposed{<:MtlArray}, src::Exposed{<:MtlArray,<:Base.ReshapedArray}
)
  return copyto!(dest, expose(parent(src)))
end
