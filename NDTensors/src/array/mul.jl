function LinearAlgebra.mul!(CM::Unwrap.Exposed{<:Array}, AM::Unwrap.Exposed{<:Array}, BM::Unwrap.Exposed{<:Array}, α, β)
  @strided mul!(CM.object, AM.object, BM.object, α, β)
  return CM.object
end
