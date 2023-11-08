using Test
using NDTensors.Unwrap
using LinearAlgebra

## Still working on this
## TODO add cu
@testset "Testing Unwrap" begin
  v = Vector{Float64}(undef, 10)
  vt = transpose(v)
  va = v'

  E = expose(v)
  Et = expose(vt)
  Ea = expose(va)
  @test typeof(E) == Exposed{Vector{Float64},Vector{Float64}}
  @test typeof(Et) ==
    Exposed{Vector{Float64},LinearAlgebra.Transpose{Float64,Vector{Float64}}}
  @test typeof(Ea) ==
    Exposed{Vector{Float64},LinearAlgebra.Adjoint{Float64,Vector{Float64}}}

  @test parent(E) == v
  @test parent(Et) == v
  @test parent(Ea) == v
  @test transpose(E) == vt
  @test cpu(E) == v
  @test cpu(Et) == vt

  m = reshape(v, (5, 2))
  mt = transpose(m)
  ma = m'
  E = expose(m)
  Et = expose(mt)
  Ea = expose(ma)

  @test typeof(E) == Exposed{Matrix{Float64},Matrix{Float64}}
  @test typeof(Et) ==
    Exposed{Matrix{Float64},LinearAlgebra.Transpose{Float64,Matrix{Float64}}}
  @test typeof(Ea) ==
    Exposed{Matrix{Float64},LinearAlgebra.Adjoint{Float64,Matrix{Float64}}}

  o = Vector{Float32}(undef, 1)
  expose(o)[] = 2
  expose(o)[] == 2

  fill!(m, 0)
  @test any(!Base.isinf, expose(m))

  mp = copy(Ea)
  mp == ma
  fill!(ma, 2.0)
  copyto!(expose(mp), expose(ma))
  mp == ma

  q, r = qr(expose(mp))
  @test q * r ≈ mp

  q, r = Unwrap.qr_positive(expose(mp))
  @test q * r ≈ mp

  square = rand(Float64, (10, 10))
  ## TODO finish this test
  l, U = eigen(expose(square))

  U, S, V, = svd(expose(mp))
  @test U * Diagonal(S) * V' ≈ mp

  cm = fill!(Matrix{Float64}(undef, (2, 2)), 0.0)
  mul!(expose(cm), expose(mp), expose(mp'), 1.0, 0.0)
  cm ≈ mp * mp'

  @test permutedims(expose(mp), (2, 1)) == transpose(mp)
  fill!(mt, 3.0)
  permutedims!(expose(m), expose(mt), (2, 1))
  @test norm(m) == sqrt(3^2 * 10)
  @test size(m) == (5, 2)
  permutedims!(expose(m), expose(mt), (2, 1), +)
  @test size(m) == (5, 2)
  @test norm(m) == sqrt(6^2 * 10)
end
