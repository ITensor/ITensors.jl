## Code adapted from NDTensors/ext/NDTensorsCUDAExt/copyto.jl

# Same definition as `MtlArray`.
function Base.copy(src::Exposed{<:ROCArray,<:Base.ReshapedArray})
  return reshape(copy(parent(src)), size(unexpose(src)))
end

function Base.copy(
  src::Exposed{
    <:ROCArray,<:SubArray{<:Any,<:Any,<:Base.ReshapedArray{<:Any,<:Any,<:Adjoint}}
  },
)
  return copy(@view copy(expose(parent(src)))[parentindices(unexpose(src))...])
end
