function LinearAlgebra.mul!(
  CM::Exposed{<:Array},
  AM::Exposed{<:Array},
  BM::Exposed{<:Array},
  α,
  β,
)
  @strided mul!(CM.object, AM.object, BM.object, α, β)
  return CM.object
end
