using Test
using NDTensors.Unwrap
using NDTensors
using LinearAlgebra

include("../../../test/device_list.jl")
## Still working on this
## TODO add cu
@testset "Testing Unwrap" for dev in devices_list(ARGS)
  v = dev(Vector{Float64}(undef, 10))
  vt = transpose(v)
  va = v'

  E = expose(v)
  Et = expose(vt)
  Ea = expose(va)
  v_type = typeof(v)
  e_type = eltype(v)
  @test typeof(E) == Exposed{v_type,v_type}
  @test typeof(Et) ==
    Exposed{v_type,LinearAlgebra.Transpose{e_type,v_type}}
  @test typeof(Ea) ==
    Exposed{v_type,LinearAlgebra.Adjoint{e_type,v_type}}

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

  m_type = typeof(m)
  @test typeof(E) == Exposed{m_type, m_type}
  @test typeof(Et) ==
    Exposed{m_type, LinearAlgebra.Transpose{e_type,m_type}}
  @test typeof(Ea) ==
    Exposed{m_type, LinearAlgebra.Adjoint{e_type,m_type}}

  o = dev(Vector{Float32})(undef, 1)
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

  square = NDTensors.cu(rand(Float64, (10, 10)))
  square = (square + square) ./ 2.
  ## CUDA only supports Hermitian or Symmetric eigen decompositions
  ## So I symmetrize square and call symetric here
  l, U = eigen(expose(Symmetric(square)))

  U, S, V, = svd(expose(mp))
  @test U * Diagonal(S) * V' ≈ mp

  cm = dev(fill!(Matrix{Float64}(undef, (2, 2)), 0.0))
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
  
  m = reshape(m, (5,2,1))
  mt = fill!(similar(m), 3.)
  m = permutedims(expose(m), (2,1,3))
  @test size(m) == (2,5,1)
  permutedims!(expose(m), expose(mt), (2,1,3))
  @test norm(m) == sqrt(3^2 * 10)
  permutedims!(expose(m), expose(mt), (2,1,3), -)
  @test norm(m) == 0
end
