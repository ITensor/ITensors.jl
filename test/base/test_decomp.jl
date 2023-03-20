using ITensors, LinearAlgebra, Test

#
#  Decide if rank 2 tensor is upper triangular, i.e. all zeros below the diagonal.
#
function is_upper(At::NDTensors.Tensor)::Bool
  nr, nc = dims(At)
  dc = Base.max(0, dim(nr) - dim(nc)) #column off set for rectangular matrices.
  nzeros = 0
  for i in CartesianIndices(At)
    if i[1] > i[2] + dc
      if abs(At[i]) > 0.0 #row>col is lower triangle
        return false
      else
        nzeros += 1
      end
    end
  end
  #
  #  Debug code:  Make some noise if At is not a vector and we still found no zeros.
  #
  # if nzeros==0 && nr>1 && nc>1 
  #   @show nr nc dc At
  # end
  return true
end

#
#  A must be rank 2
#
function is_upper(l::Index, A::ITensor, r::Index)::Bool
  @assert length(inds(A)) == 2
  if inds(A) != IndexSet(l, r)
    A = permute(A, l, r)
  end
  return is_upper(NDTensors.tensor(A))
end

#
#  With left index specified
#
function is_upper(l::Index, A::ITensor)::Bool
  other = noncommoninds(A, l)
  if (length(other) == 1)
    return is_upper(l, A, other[1])
  else
    # use combiner to gather all the "other" indices into one.
    C = combiner(other...)
    AC = A * C
    return is_upper(l, AC, combinedind(C))
  end
end
is_lower(l::Index, A::ITensor)::Bool = is_upper(A, l)

#
#  With right index specified
#
function is_upper(A::ITensor, r::Index)::Bool
  other = noncommoninds(A, r)
  if (length(other) == 1)
    return is_upper(other[1], A, r)
  else
    C = combiner(other...)
    AC = A * C
    return is_upper(combinedind(C), AC, r)
  end
end
is_lower(A::ITensor, r::Index)::Bool = is_upper(r, A)

function diag_upper(l::Index, A::ITensor)
  At = NDTensors.tensor(A * combiner(noncommoninds(A, l)...))
  if size(At) == (1,)
    return At
  end
  @assert length(size(At)) == 2
  return diag(At)
end

function diag_lower(l::Index, A::ITensor)
  At = NDTensors.tensor(A * combiner(noncommoninds(A, l)...)) #render down ot order 2
  if size(At) == (1,)
    return At
  end
  @assert length(size(At)) == 2
  nr, nc = size(At)
  dc = Base.max(0, nc - nr) #diag starts dc+1 columns out from the left
  At1 = At[:, (dc + 1):nc] #chop out the first dc columns
  return diag(At1) #now we can use the stock diag function.
end

@testset "ITensor Decompositions" begin
  @testset "truncate!" begin
    a = [0.1, 0.01, 1e-13]
    @test NDTensors.truncate!(a; use_absolute_cutoff=true, cutoff=1e-5) ==
      (1e-13, (0.01 + 1e-13) / 2)
    @test length(a) == 2

    # Negative definite spectrum treated by taking 
    # square (if singular values) or absolute values
    a = [-0.12, -0.1]
    @test NDTensors.truncate!(a) == (0.0, 0.0)
    @test length(a) == 2

    a = [-0.1, -0.01, -1e-13]
    @test NDTensors.truncate!(a; use_absolute_cutoff=true, cutoff=1e-5) ==
      (1e-13, (0.01 + 1e-13) / 2)
    @test length(a) == 2
  end

  @testset "factorize" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, j)
    @test_throws ErrorException factorize(A, i; dir="left")
    @test_throws ErrorException factorize(A, i; ortho="fakedir")
  end

  @testset "factorize with eigen_perturbation" begin
    l = Index(4, "l")
    s1 = Index(2, "s1")
    s2 = Index(2, "s2")
    r = Index(4, "r")

    phi = randomITensor(l, s1, s2, r)

    drho = randomITensor(l', s1', l, s1)
    drho += swapprime(drho, 0, 1)
    drho .*= 1E-5

    U, B = factorize(phi, (l, s1); ortho="left", eigen_perturbation=drho)
    @test norm(U * B - phi) < 1E-5

    # Not allowed to use eigen_perturbation with which_decomp
    # other than "automatic" or "eigen":
    @test_throws ErrorException factorize(
      phi, (l, s1); ortho="left", eigen_perturbation=drho, which_decomp="svd"
    )
  end

  @testset "QR/RQ/QL/LQ decomp on MPS dense $elt tensor with all possible collections on Q/R/L" for ninds in
                                                                                                    [
      0, 1, 2, 3
    ],
    elt in [Float64, ComplexF64]

    l = Index(5, "l")
    s = Index(2, "s")
    r = Index(5, "r")
    A = randomITensor(elt, l, s, r)
    Ainds = inds(A)
    Linds = Ainds[1:ninds]
    Rinds = uniqueinds(A, Linds...)
    Q, R, q = qr(A, Linds) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(R)) == 3 - ninds + 1
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, R)
    @test hastags(q, "Link,qr")
    if (length(inds(R)) > 1)
      @test is_upper(q, R) #specify the left index
    end
    Q1, R1, q1 = qr(A, Linds, Rinds; tags="Link,myqr") #make sure the same call with both L & R indices give the same answer.
    Q1 = replaceind(Q1, q1, q)
    R1 = replaceind(R1, q1, q)
    @test norm(Q - Q1) == 0.0
    @test norm(R - R1) == 0.0
    @test hastags(q1, "Link,myqr")

    R, Q, q = rq(A, Linds)
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test A ≈ Q * R atol = 1e-13 #With ITensors R*Q==Q*R
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, R)
    @test hastags(q, "rq")
    if (length(inds(R)) > 1)
      @test is_upper(R, q) #specify the right index
    end
    R1, Q1, q1 = rq(A, Linds, Rinds; tags="Link,myrq") #make sure the same call with both L & R indices give the same answer.
    Q1 = replaceind(Q1, q1, q)
    R1 = replaceind(R1, q1, q)
    @test norm(Q - Q1) == 0.0
    @test norm(R - R1) == 0.0
    @test hastags(q1, "myrq")
    @test hastags(q1, "Link")

    L, Q, q = lq(A, Linds)
    @test length(inds(L)) == ninds + 1 #+1 to account for new lq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test A ≈ Q * L atol = 1e-13 #With ITensors L*Q==Q*L
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, L)
    @test hastags(q, "lq")
    if (length(inds(L)) > 1)
      @test is_lower(L, q) #specify the right index
    end
    L1, Q1, q1 = lq(A, Linds, Rinds; tags="Link,mylq") #make sure the same call with both L & R indices give the same answer.
    Q1 = replaceind(Q1, q1, q)
    L1 = replaceind(L1, q1, q)
    @test norm(Q - Q1) == 0.0
    @test norm(L - L1) == 0.0
    @test hastags(q1, "mylq")
    @test hastags(q1, "Link")

    Q, L, q = ql(A, Linds)
    @test length(inds(Q)) == ninds + 1 #+1 to account for new lq,Link index.
    @test length(inds(L)) == 3 - ninds + 1
    @test A ≈ Q * L atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, L)
    @test hastags(q, "ql")
    if (length(inds(L)) > 1)
      @test is_lower(q, L) #specify the right index
    end
    Q1, L1, q1 = ql(A, Linds, Rinds; tags="Link,myql") #make sure the same call with both L & R indices give the same answer.
    Q1 = replaceind(Q1, q1, q)
    L1 = replaceind(L1, q1, q)
    @test norm(Q - Q1) == 0.0
    @test norm(L - L1) == 0.0
    @test hastags(q1, "myql")
    @test hastags(q1, "Link")
  end

  @testset "QR/RQ dense on MP0 tensor with all possible collections on Q,R" for ninds in [
    0, 1, 2, 3, 4
  ]
    l = Index(5, "l")
    s = Index(2, "s")
    r = Index(10, "r")
    A = randomITensor(l, s, s', r)
    Ainds = inds(A)
    Q, R, q = qr(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(R)) == 4 - ninds + 1
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13

    R, Q, q = rq(A, Ainds[1:ninds])
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 4 - ninds + 1
    @test A ≈ Q * R atol = 1e-13 #With ITensors R*Q==Q*R
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
  end

  @testset "QR/RQ block sparse on MPS tensor with all possible collections on Q,R" for ninds in
                                                                                       [
    0, 1, 2, 3
  ]
    expected_Qflux = [QN(), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0)]
    expected_RLflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", -0), QN()]
    l = dag(Index(QN("Sz", 0) => 1, QN("Sz", 1) => 1, QN("Sz", -1) => 1; tags="l"))
    s = Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags="s")
    r = Index(QN("Sz", 0) => 1, QN("Sz", 1) => 1, QN("Sz", -1) => 1; tags="r")
    A = randomITensor(l, s, r)
    @test flux(A) == QN("Sz", 0)
    Ainds = inds(A)
    Q, R, q = qr(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(R)) == 3 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(R) == expected_RLflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
    # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
    # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    Q, L, q = ql(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(L)) == 3 - ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(L) == expected_RLflux[ninds + 1]
    @test A ≈ Q * L atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    R, Q, q = rq(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(R) == expected_RLflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    L, Q, q = lq(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(L)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(L) == expected_RLflux[ninds + 1]
    @test A ≈ Q * L atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
  end

  @testset "QR/QL block sparse on MPO tensor with all possible collections on Q,R" for ninds in
                                                                                       [
    0, 1, 2, 3, 4
  ]
    expected_Qflux = [QN(), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0)]
    expected_RLflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN()]
    l = dag(Index(QN("Sz", 0) => 3; tags="l"))
    s = Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags="s")
    r = Index(QN("Sz", 0) => 3; tags="r")
    A = randomITensor(l, s, dag(s'), r)
    @test flux(A) == QN("Sz", 0)
    Ainds = inds(A)
    Q, R, q = qr(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(R)) == 4 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(R) == expected_RLflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
    # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
    # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    Q, L, q = ql(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(L)) == 4 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(L) == expected_RLflux[ninds + 1]
    @test A ≈ Q * L atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
  end

  @testset "QR/QL/RQ/LQ dense with positive R" for ninds in [0, 1, 2, 3]
    l = Index(3, "l")
    s = Index(5, "s")
    r = Index(7, "r")
    A = randomITensor(l, s, s', r)
    Ainds = inds(A)

    Q, R, q = qr(A, Ainds[1:ninds]; positive=true)
    @test min(diag_upper(q, R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    Q, L, q = ql(A, Ainds[1:ninds]; positive=true)
    @test min(diag_lower(q, L)...) > 0.0
    @test A ≈ Q * L atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13

    R, Q, q = rq(A, Ainds[1:ninds]; positive=true)
    @test min(diag_lower(q, R)...) > 0.0 #transpose R is lower
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    L, Q, q = lq(A, Ainds[1:ninds]; positive=true)
    @test min(diag_upper(q, L)...) > 0.0 #transpose L is upper
    @test A ≈ Q * L atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
  end

  @testset "QR/QL block sparse with positive R" begin
    l = dag(Index(QN("Sz", 0) => 3; tags="l"))
    s = Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags="s")
    r = Index(QN("Sz", 0) => 3; tags="r")
    A = randomITensor(l, s, dag(s'), r)
    Q, R, q = qr(A, l, s, dag(s'); positive=true)
    @test min(diag(R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
    Q, L, q = ql(A, l, s, dag(s'); positive=true)
    @test min(diag(L)...) > 0.0
    @test A ≈ Q * L atol = 1e-13
  end

  @testset "factorize with QR" begin
    l = Index(5, "l")
    s = Index(2, "s")
    r = Index(10, "r")
    A = randomITensor(l, s, r)
    Q, R, = factorize(A, l, s; which_decomp="qr")
    q = commonind(Q, R)
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13

    R, Q, = factorize(A, l, s; which_decomp="qr", ortho="right")
    q = commonind(Q, R)
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
  end

  @testset "eigen" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, i')
    eigA = eigen(A)
    Dt, Ut = eigen(NDTensors.tensor(A))
    eigArr = eigen(array(A))
    @test diag(array(eigA.D), 0) ≈ eigArr.values
    @test diag(array(Dt), 0) == eigArr.values

    @test_throws ArgumentError eigen(ITensor(NaN, i', i))
    @test_throws ArgumentError eigen(ITensor(NaN, i', i); ishermitian=true)
    @test_throws ArgumentError eigen(ITensor(complex(NaN), i', i))
    @test_throws ArgumentError eigen(ITensor(complex(NaN), i', i); ishermitian=true)
    @test_throws ArgumentError eigen(ITensor(Inf, i', i))
    @test_throws ArgumentError eigen(ITensor(Inf, i', i); ishermitian=true)
    @test_throws ArgumentError eigen(ITensor(complex(Inf), i', i))
    @test_throws ArgumentError eigen(ITensor(complex(Inf), i', i); ishermitian=true)
  end

  @testset "exp function" begin
    At = rand(10, 10)
    k = Index(10, "k")
    A = itensor(At + transpose(At), k, k')
    @test Array(exp(Hermitian(NDTensors.tensor(A)))) ≈ exp(At + transpose(At))
  end

  @testset "Spectrum" begin
    i = Index(100, "i")
    j = Index(100, "j")

    U, S, V = svd(rand(100, 100))
    S ./= norm(S)
    A = itensor(U * ITensors.diagm(0 => S) * V', i, j)

    spec = svd(A, i).spec

    @test eigs(spec) ≈ S .^ 2
    @test truncerror(spec) == 0.0

    spec = svd(A, i; maxdim=length(S) - 3).spec
    @test truncerror(spec) ≈ sum(S[(end - 2):end] .^ 2)

    @test entropy(Spectrum([0.5; 0.5], 0.0)) == log(2)
    @test entropy(Spectrum([1.0], 0.0)) == 0.0
    @test entropy(Spectrum([0.0], 0.0)) == 0.0

    @test isnothing(eigs(Spectrum(nothing, 1.0)))
    @test_throws ErrorException entropy(Spectrum(nothing, 1.0))
    @test truncerror(Spectrum(nothing, 1.0)) == 1.0
  end

  @testset "Eigen QN flux regression test" begin
    cutoff = 1E-12
    N = 4
    s = siteinds("S=1", N; conserve_qns=true)
    A = randomITensor(QN("Sz", 2), s[1], s[2], s[3])

    R = A * dag(prime(A, s[1], s[2]))
    F = eigen(R, (s[1], s[2]), (s[1]', s[2]'))

    @test flux(F.Vt) == QN("Sz", 0)
  end

  @testset "SVD block_mindim keyword" begin
    i = Index(
      [
        QN("Sz", 4) => 1,
        QN("Sz", 2) => 4,
        QN("Sz", 0) => 6,
        QN("Sz", -2) => 4,
        QN("Sz", -4) => 1,
      ],
      "i",
    )
    j = sim(i)
    X = randomITensor(QN("Sz", 0), i, j)

    min_blockdim = 2
    U, S, V = svd(X, i; cutoff=1E-1, min_blockdim)
    u = commonind(S, U)

    @test nblocks(u) == nblocks(i)
    for b in 1:nblocks(u)
      @test blockdim(u, b) == blockdim(i, b) || blockdim(u, b) >= min_blockdim
    end
  end
end

nothing
