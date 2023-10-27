function mul!!(CM::AbstractArray, AM::AbstractArray, BM::AbstractArray, α, β)
  CM = LinearAlgebra.mul!(expose(CM), expose(AM), expose(BM), α, β)
  return CM
end
