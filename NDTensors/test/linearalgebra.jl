using NDTensors
using LinearAlgebra
using Test

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

@testset "Dense $qx decomposition, elt=$elt, positve=$positive, singular=$singular, rank_reveal=$rank_reveal" for qx in
                                                                                        [
    qr, ql
  ],
  elt in [Float64, ComplexF64, Float32, ComplexF32],
  positive in [false, true],
  singular in [false, true],
  rank_reveal in [false,true],

  eps = Base.eps(real(elt)) * 30 #this is set rather tight, so if you increase/change m,n you may have open up the tolerance on eps.
  rr_cutoff = rank_reveal ? eps*1.0 : -1.0
  n, m = 4, 8
  #
  # Wide matrix (more columns than rows)
  #
  A = randomTensor(elt, (n, m))
  # We want to test 0.0 on the diagonal.  We need make all rows linearly dependent 
  # gaurantee this with numerical roundoff.
  if singular
    for i in 2:n
      A[i, :] = A[1, :]*1.05^n
    end
  end
  Q, X = qx(A; positive=positive, rr_cutoff=rr_cutoff) #X is R or L.
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, dim(Q, 2))) atol = eps
  if dim(Q, 1)==dim(Q, 2)
    @test array(Q) * array(Q)' ≈ Diagonal(fill(1.0, min(n, m))) atol = eps
  end
  if positive
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  if rr_cutoff>0 && singular
    @test dim(Q, 2)==1 #make sure the rank revealing mechanism hacked off the columns of Q (and rows of X).
    @test dim(X ,1)==1 #Redundant?
   end
  #
  # Tall matrix (more rows than cols)
  #
  A = randomTensor(elt, (m, n)) #Tall array
  # We want to test 0.0 on the diagonal.  We need make all rows equal to gaurantee this with numerical roundoff.
  if singular
    for i in 2:m
      A[i, :] = A[1, :]
    end
  end
  Q, X = qx(A; positive=positive, rr_cutoff=rr_cutoff)
  @test A ≈ Q * X atol = eps
  @test array(Q)' * array(Q) ≈ Diagonal(fill(1.0, dim(Q, 2))) atol = eps
  #@test array(Q) * array(Q)' no such relationship for tall matrices.
  if positive
    nr, nc = size(X)
    dr = qx == ql ? Base.max(0, nc - nr) : 0
    diagX = diag(X[:, (1 + dr):end]) #location of diag(L) is shifted dr columns over the right.
    @test all(real(diagX) .>= 0.0)
    @test all(imag(diagX) .== 0.0)
  end
  if rr_cutoff>0 && singular
    @test dim(Q, 2)==1 #make sure the rank revealing mechanism hacked off the columns of Q (and rows of X).
    @test dim(X ,1)==1 #Redundant?
  end
end

nothing
