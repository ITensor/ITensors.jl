@eval module $(gensym())
using NDTensors
using LinearAlgebra: Diagonal, qr, diag
using Test: @testset, @test
using GPUArraysCore: @allowscalar
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: NDTensorsTestUtils

@testset "random_orthog" begin
  n, m = 10, 4
  O1 = random_orthog(n, m)
  @test eltype(O1) == Float64
  @test norm(transpose(O1) * O1 - Diagonal(fill(1.0, m))) < 1E-14
  O2 = random_orthog(m, n)
  @test norm(O2 * transpose(O2) - Diagonal(fill(1.0, m))) < 1E-14
end

@testset "random_unitary" begin
  n, m = 10, 4
  U1 = random_unitary(n, m)
  @test eltype(U1) == ComplexF64
  @test norm(U1' * U1 - Diagonal(fill(1.0, m))) < 1E-14
  U2 = random_unitary(m, n)
  @test norm(U2 * U2' - Diagonal(fill(1.0, m))) < 1E-14
end

@testset "QX testing" begin
  @testset "Dense $qx decomposition, elt=$elt, positve=$positive, singular=$singular, device=$dev" for qx in
                                                                                                       [
      qr, ql
    ],
    elt in (Float64, ComplexF64, Float32, ComplexF32),
    positive in [false, true],
    singular in [false, true],
    dev in NDTensorsTestUtils.devices_list(copy(ARGS))

    ## Skip Float64 on Metal
    if !NDTensorsTestUtils.is_supported_eltype(dev, elt)
      continue
    end
    eps = Base.eps(real(elt)) * 100 #this is set rather tight, so if you increase/change m,n you may have open up the tolerance on eps.
    n, m = 4, 8
    Id = Diagonal(fill(1.0, min(n, m)))
    #
    # Wide matrix (more columns than rows)
    #
    A = dev(randomTensor(elt, (n, m)))
    # We want to test 0.0 on the diagonal.  We need to make all rows equal to gaurantee this with numerical roundoff.
    @allowscalar if singular
      for i in 2:n
        A[i, :] = A[1, :]
      end
    end
    Q, X = qx(A; positive=positive) #X is R or L.
    Ap = Q * X
    @test NDTensors.cpu(A) ≈ NDTensors.cpu(Ap) atol = eps
    @test NDTensors.cpu(array(Q)' * array(Q)) ≈ Id atol = eps
    @test NDTensors.cpu(array(Q) * array(Q)') ≈ Id atol = eps
    @allowscalar if positive
      nr, nc = size(X)
      dr = qx == ql ? Base.max(0, nc - nr) : 0
      diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
      @test all(real(diagX) .>= 0.0)
      @test all(imag(diagX) .== 0.0)
    end
    #
    # Tall matrix (more rows than cols)
    #
    A = dev(randomTensor(elt, (m, n))) #Tall array
    # We want to test 0.0 on the diagonal.  We need make all rows equal to gaurantee this with numerical roundoff.
    @allowscalar if singular
      for i in 2:m
        A[i, :] = A[1, :]
      end
    end
    Q, X = qx(A; positive=positive)
    Ap = Q * X
    @test NDTensors.cpu(A) ≈ NDTensors.cpu(Ap) atol = eps
    @test NDTensors.cpu(array(Q)' * array(Q)) ≈ Id atol = eps
    @allowscalar if positive
      nr, nc = size(X)
      dr = qx == ql ? Base.max(0, nc - nr) : 0
      diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
      @test all(real(diagX) .>= 0.0)
      @test all(imag(diagX) .== 0.0)
    end
  end
end

nothing
end
