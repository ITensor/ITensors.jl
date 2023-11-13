using Test
using NDTensors.Unwrap
using NDTensors
using LinearAlgebra
using GPUArraysCore: @allowscalar

include("../../../test/device_list.jl")
@testset "Testing Unwrap $dev, $elt" for dev in devices_list(ARGS),
  elt in (Float32, ComplexF32)

  v = dev(randn(elt, 10))
  vt = transpose(v)
  va = v'

  E = expose(v)
  Et = expose(vt)
  Ea = expose(va)
  v_type = typeof(v)
  e_type = eltype(v)
  @test typeof(E) == Exposed{v_type,v_type}
  @test typeof(Et) == Exposed{v_type,LinearAlgebra.Transpose{e_type,v_type}}
  @test typeof(Ea) == Exposed{v_type,LinearAlgebra.Adjoint{e_type,v_type}}

  @test parent(E) == v
  @test parent(Et) == v
  @test parent(Ea) == v
  @test transpose(E) == vt
  @test cpu(E) == cpu(v)
  @test cpu(Et) == cpu(vt)

  m = reshape(v, (5, 2))
  mt = transpose(m)
  ma = m'
  E = expose(m)
  Et = expose(mt)
  Ea = expose(ma)

  m_type = typeof(m)
  @test typeof(E) == Exposed{m_type,m_type}
  @test typeof(Et) == Exposed{m_type,LinearAlgebra.Transpose{e_type,m_type}}
  @test typeof(Ea) == Exposed{m_type,LinearAlgebra.Adjoint{e_type,m_type}}

  o = dev(randn(elt, 1))
  expose(o)[] = 2
  @test expose(o)[] == 2

  fill!(m, 0)
  @test any(!Base.isinf, expose(m))

  mp = copy(Ea)
  @test mp == ma
  fill!(ma, 2.0)
  copyto!(expose(mp), expose(ma))
  @test mp == ma

  q, r = qr(expose(mp))
  @test q * r ≈ mp

  q, r = Unwrap.qr_positive(expose(mp))
  @test q * r ≈ mp

  square = dev(rand(real(elt), (10, 10)))
  square = (square + transpose(square)) / 2
  ## CUDA only supports Hermitian or Symmetric eigen decompositions
  ## So I symmetrize square and call symetric here
  l, U = eigen(expose(Symmetric(square)))
  @test eltype(l) == real(elt)
  @test eltype(U) == real(elt)
  @test square * U ≈ U * Diagonal(l)

  square = dev(rand(elt, (10, 10)))
  # Can use `hermitianpart` in Julia 1.10
  square = (square + square') / 2
  ## CUDA only supports Hermitian or Symmetric eigen decompositions
  ## So I symmetrize square and call symetric here
  l, U = eigen(expose(Hermitian(square)))
  @test eltype(l) == real(elt)
  @test eltype(U) == elt
  @test square * U ≈ U * Diagonal(l)

  U, S, V, = svd(expose(mp))
  @test eltype(U) == elt
  @test eltype(S) == real(elt)
  @test eltype(V) == elt
  @test U * Diagonal(S) * V' ≈ mp

  cm = dev(randn(elt, 2, 2))
  mul!(expose(cm), expose(mp), expose(mp'), 1.0, 0.0)
  @test cm ≈ mp * mp'

  @test permutedims(expose(mp), (2, 1)) == transpose(mp)
  fill!(mt, 3)
  permutedims!(expose(m), expose(mt), (2, 1))
  @test norm(m) ≈ sqrt(3^2 * 10)
  @test size(m) == (5, 2)
  permutedims!(expose(m), expose(mt), (2, 1), +)
  @test size(m) == (5, 2)
  @test norm(m) ≈ sqrt(6^2 * 10)

  m = reshape(m, (5, 2, 1))
  mt = fill!(similar(m), 3.0)
  m = permutedims(expose(m), (2, 1, 3))
  @test size(m) == (2, 5, 1)
  permutedims!(expose(m), expose(mt), (2, 1, 3))
  @test norm(m) ≈ sqrt(3^2 * 10)
  permutedims!(expose(m), expose(mt), (2, 1, 3), -)
  @test norm(m) == 0

  x = dev(rand(elt, 4, 4))
  y = dev(rand(elt, 4, 4))
  copyto!(expose(y), expose(x))
  @test y == x

  y = dev(rand(elt, 4, 4))
  x = Base.ReshapedArray(dev(rand(elt, 16)), (4, 4), ())
  copyto!(expose(y), expose(x))
  @test NDTensors.cpu(y) == NDTensors.cpu(x)
  @test NDTensors.cpu(copy(expose(x))) == NDTensors.cpu(x)

  y = dev(rand(elt, 4, 4))
  x = @view dev(rand(elt, 8, 8))[1:4, 1:4]
  copyto!(expose(y), expose(x))
  @test y == x
  @test copy(x) == x

  y = dev(randn(elt, 16))
  x = reshape(dev(randn(elt, 4, 4))', 16)
  copyto!(expose(y), expose(x))
  @allowscalar begin
    @test y == x
    @test copy(x) == x
  end

  y = dev(randn(elt, 8))
  x = @view reshape(dev(randn(elt, 8, 8))', 64)[1:8]
  copyto!(expose(y), expose(x))
  @allowscalar begin  
    @test y == x
    @test copy(x) == x
  end

  y = Base.ReshapedArray(dev(randn(elt, 16)), (4, 4), ())
  x = dev(randn(elt, 4, 4))
  permutedims!(expose(y), expose(x), (2, 1))
  @test NDTensors.cpu(y) == transpose(NDTensors.cpu(x))

  ##########################################
  ### Testing an issue with CUDA&Metal transpose/adjoint mul
  A = dev(randn(elt, (3, 2)))
  B = dev(randn(elt, (3, 4)))
  C = dev(randn(elt, (4, 2)))
  Cp = copy(C)

  ## This fails with scalar indexing
  if dev != NDTensors.cpu
    @test_broken mul!(transpose(C), transpose(A), B, true, false)
  end
  mul!(C, transpose(B), A, true, false)
  mul!(expose(transpose(Cp)), expose(transpose(A)), expose(B), true, false)
  @test C ≈ Cp
  Cp = zero(C)
  ## Try calling mul!! with transposes to verify that code works
  Cpt = NDTensors.mul!!(transpose(Cp), transpose(A), B, true, false)
  @test transpose(Cpt) ≈ C

  Cp = zero(C)
  ## This fails with scalar indexing 
  if dev != NDTensors.cpu
    @test_broken mul!(C', A', B, true, false)
  end
  mul!(C, B', A, true, false)
  mul!(expose(Cp'), expose(A'), expose(B), true, false)
  @test C ≈ Cp
  Cp = zero(C)
  Cpt = NDTensors.mul!!(Cp', A', B, true, false)
  @test Cpt' ≈ C
end
