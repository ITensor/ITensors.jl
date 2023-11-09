function LinearAlgebra.mul!(
  C::Exposed{<:CuArray,<:LinearAlgebra.Transpose},
  A::Exposed{<:CuArray},
  B::Exposed{<:CuArray},
  α,
  β,
)
  return mul!(
    expose(parent(C)), expose(transpose(unexpose(B))), expose(transpose(unexpose(A))), α, β
  )
end
