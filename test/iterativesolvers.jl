using ITensors
using Test

# Wrap an ITensor with pairs of primed and
# unprimed indices to pass to davidson
struct ITensorMap
  A::ITensor
end
Base.eltype(M::ITensorMap) = eltype(M.A)
Base.size(M::ITensorMap) = dim(IndexSet(filterinds(M.A; plev=0)...))
(M::ITensorMap)(v::ITensor) = noprime(M.A * v)

@testset "Complex davidson" begin
  d = 10
  i = Index(d, "i")
  A = randomITensor(ComplexF64, i, i')
  A = mapprime(A * mapprime(dag(A), 0 => 2), 2 => 1)
  M = ITensorMap(A)

  v = randomITensor(i)
  λ, v = davidson(M, v; maxiter=10)
  @test M(v) ≈ λ * v

  v = randomITensor(ComplexF64, i)
  λ, v = davidson(M, v; maxiter=10)
  @test M(v) ≈ λ * v
end

nothing
