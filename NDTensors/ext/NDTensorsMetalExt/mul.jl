# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray,<:Transpose},
  AM::Exposed{<:MtlArray},
  BM::Exposed{<:MtlArray},
  α,
  β,
)
  mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
  return unexpose(CM)
end

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray,<:Adjoint}, AM::Exposed{<:MtlArray}, BM::Exposed{<:MtlArray}, α, β
)
  mul!(CM', BM', AM', α, β)
  return unexpose(CM)
end

## Fix issue in Metal.jl where it cannot distinguish Transpose{Reshape{Adjoint{CuArray}}}
## as a CuArray and calls generic matmul
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray},
  AM::Exposed{<:MtlArray},
  BM::Exposed{
    <:MtlArray,
    <:LinearAlgebra.Transpose{
      <:Any,<:Base.ReshapedArray{<:Any,<:Any,<:LinearAlgebra.Adjoint}
    },
  },
  α,
  β,
)
  mul!(CM, AM, expose(transpose(copy(expose(parent(BM))))), α, β)
  return unexpose(CM)
end
