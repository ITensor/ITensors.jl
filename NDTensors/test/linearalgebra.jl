using NDTensors
using LinearAlgebra
using Test

# Not available on CI machine that tests NDTensors.
# using Random
# Random.seed!(314159)

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

@testset "Dense $qx decomposition, elt=$elt, positve=$positive, singular=$singular, rank_reveal=$rank_reveal, pivot=$pivot" for qx in
                                                                                                                                [
    qr, ql
  ],
  elt in [Float64, ComplexF64, Float32, ComplexF32],
  positive in [false, true],
  singular in [false, true],
  rank_reveal in [false, true],
  pivot in [false, true],
  return_Rp in [false, true]

  if qx == ql && (rank_reveal || pivot)
    continue
  end

  # avoid warnings.
  if !(rank_reveal || pivot) && return_Rp
    continue
  end

  eps = Base.eps(real(elt)) * 30
  #this is set rather tight, so if you increase/change m,n you may have open up the tolerance on eps.
  atol = rank_reveal ? eps * 1.0 : -1.0
  n, m = 4, 8
  #
  # Wide matrix (more columns than rows)
  #
  A = randomTensor(elt, (n, m))
  # We want to test 0.0 on the diagonal.  We need make all rows linearly dependent 
  # gaurantee this with numerical roundoff.
  if singular
    for i in 2:3
      A[i, :] = A[1, :] * 1.05^n
    end
  end
  # you can set verbose=true if you want to get debug output on rank reduction.
  Q, X, Xp = qx(
    A; positive=positive, atol=atol, pivot=pivot, return_Rp=return_Rp, verbose=false
  ) #X is R or L. 
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, dim(Q, 2))) atol = eps
  if dim(Q, 1) == dim(Q, 2)
    @test array(Q) * array(Q)' ≈ Diagonal(fill(1.0, min(n, m))) atol = eps
  end
  if positive && !rank_reveal && !pivot
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  if positive && !isnothing(Xp)
    nr, nc = size(Xp)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(Xp[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end

  if atol >= 0 && singular
    @test dim(Q, 2) == 2 #make sure the rank revealing mechanism hacked off the columns of Q (and rows of X).
    @test dim(X, 1) == 2 #Redundant?
  end
  if (atol >= 0.0 || pivot) && qx == qr
    @test !isnothing(Xp) == return_Rp
  end
  #
  # Tall matrix (more rows than cols)
  #
  A = randomTensor(elt, (m, n)) #Tall array
  # We want to test 0.0 on the diagonal.  We need make all rows equal to gaurantee this with numerical roundoff.
  if singular
    for i in 2:4
      A[i, :] = A[1, :]
    end
  end
  Q, X, Xp = qx(
    A; positive=positive, atol=atol, pivot=pivot, return_Rp=return_Rp, verbose=false
  )
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, dim(Q, 2))) atol = eps
  #@test array(Q) * array(Q)' no such relationship for tall matrices.
  if positive && !rank_reveal && !pivot
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  if positive && !isnothing(Xp)
    nr, nc = size(Xp)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(Xp[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  if atol > 0 && singular
    @test dim(Q, 2) == 4 #make sure the rank revealing mechanism hacked off the columns of Q (and rows of X).
    @test dim(X, 1) == 4 #Redundant?
  end
  if (atol >= 0.0 || pivot) && qx == qr
    @test !isnothing(Xp) == return_Rp
  end
end

nothing
