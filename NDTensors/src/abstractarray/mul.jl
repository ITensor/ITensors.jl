function mul!!(CM::AbstractArray, AM::AbstractArray, BM::AbstractArray, α, β)
  LinearAlgebra.mul!(expose(CM), expose(AM), expose(BM), α, β)
  return unexpose(CM)
end