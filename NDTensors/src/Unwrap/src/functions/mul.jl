function LinearAlgebra.mul!(CM::Exposed, AM::Exposed, BM::Exposed, α, β)
  mul!(CM.object, AM.object, BM.object, α, β)
  return CM.object
end
