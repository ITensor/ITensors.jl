using ITensors, LinearAlgebra, Test

#
#  Decide of rank 2 tensor is upper triangular, i.e. all zeros below the diagonal.
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

#
#  Build up Hamiltonians with non trival QN spaces in the link indices and further neighbour interactions.
#
function make_Heisenberg_AutoMPO(sites,NNN::Int64;J::Float64=1.0,kwargs...)::MPO
  N=length(sites)
  @assert N>=NNN
  ampo = OpSum()
  for dj=1:NNN
      f=J/dj
      for j=1:N-dj
          add!(ampo, f    ,"Sz", j, "Sz", j+dj)
          add!(ampo, f*0.5,"S+", j, "S-", j+dj)
          add!(ampo, f*0.5,"S-", j, "S+", j+dj)
      end
  end
  return MPO(ampo,sites;kwargs...)
end

function make_Hubbard_AutoMPO(sites,NNN::Int64;U::Float64=1.0,t::Float64=1.0,V::Float64=0.5,kwargs...)::MPO
  N=length(sites)
  @assert(N>=NNN)
  os = OpSum()
  for i in 1:N
    os += (U, "Nupdn", i)
  end
  for dn=1:NNN
      tj,Vj=t/dn,V/dn
      for n in 1:(N - dn)
      os += -tj, "Cdagup", n, "Cup", n + dn
      os += -tj, "Cdagup", n + dn, "Cup", n
      os += -tj, "Cdagdn", n, "Cdn", n + dn
      os += -tj, "Cdagdn", n + dn, "Cdn", n
      os +=  Vj, "Ntot"  , n, "Ntot", n + dn
      end
  end
  return MPO(os, sites;kwargs...)
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
    Q, R, q = qr(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(Q)) == ninds + 1 #+1 to account for new qr,Link index.
    @test length(inds(R)) == 3 - ninds + 1
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, R)
    @test hastags(q, "qr")
    if (length(inds(R)) > 1)
      @test is_upper(q, R) #specify the left index
    end

    R, Q, q = ITensors.rq(A, Ainds[1:ninds])
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test A ≈ Q * R atol = 1e-13 #With ITensors R*Q==Q*R
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, R)
    @test hastags(q, "rq")
    if (length(inds(R)) > 1)
      @test is_upper(R, q) #specify the right index
    end

    L, Q, q = lq(A, Ainds[1:ninds])
    @test length(inds(L)) == ninds + 1 #+1 to account for new lq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test A ≈ Q * L atol = 1e-13 #With ITensors L*Q==Q*L
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, L)
    @test hastags(q, "lq")
    if (length(inds(L)) > 1)
      @test is_lower(L, q) #specify the right index
    end

    Q, L, q = ITensors.ql(A, Ainds[1:ninds])
    @test length(inds(Q)) == ninds + 1 #+1 to account for new lq,Link index.
    @test length(inds(L)) == 3 - ninds + 1
    @test A ≈ Q * L atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    @test q == commonind(Q, L)
    @test hastags(q, "ql")
    if (length(inds(L)) > 1)
      @test is_lower(q, L) #specify the right index
    end
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

    R, Q, q = ITensors.rq(A, Ainds[1:ninds])
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
    expected_Rflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", -0), QN()]
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
    @test flux(R) == expected_Rflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
    # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
    # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
    expected_Rflux = [QN(), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0)]
    expected_Qflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN()]
    R, Q, q = ITensors.rq(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 3 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(R) == expected_Rflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
  end

  @testset "QR/RQ block sparse on MPO tensor with all possible collections on Q,R" for ninds in
                                                                                       [
    0, 1, 2, 3, 4
  ]
    expected_Qflux = [QN(), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0)]
    expected_Rflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN()]
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
    @test flux(R) == expected_Rflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
    # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
    # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

    expected_Qflux = [QN(), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0)]
    expected_Rflux = [QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN("Sz", 0), QN()]
    R, Q, q = ITensors.rq(A, Ainds[1:ninds]) #calling  qr(A) triggers not supported error.
    @test length(inds(R)) == ninds + 1 #+1 to account for new rq,Link index.
    @test length(inds(Q)) == 4 - ninds + 1
    @test flux(Q) == expected_Qflux[ninds + 1]
    @test flux(R) == expected_Rflux[ninds + 1]
    @test A ≈ Q * R atol = 1e-13
    @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
  end

  @testset "QR/RQ dense with positive R" begin
    l = Index(5, "l")
    s = Index(2, "s")
    r = Index(10, "r")
    A = randomITensor(l, s, s', r)
    Q, R, q = qr(A, l, s, s'; positive=true)
    @test min(diag(R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
    R, Q, q = ITensors.rq(A, r; positive=true)
    @test min(diag(R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
    @test Q * dag(prime(Q, q)) ≈ δ(Float64, q, q') atol = 1e-13
  end

  @testset "QR/RQ block sparse with positive R" begin
    l = dag(Index(QN("Sz", 0) => 3; tags="l"))
    s = Index(QN("Sz", -1) => 1, QN("Sz", 1) => 1; tags="s")
    r = Index(QN("Sz", 0) => 3; tags="r")
    A = randomITensor(l, s, dag(s'), r)
    Q, R, q = qr(A, l, s, s'; positive=true)
    @test min(diag(R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
    R, Q, q = ITensors.rq(A, r; positive=true)
    @test min(diag(R)...) > 0.0
    @test A ≈ Q * R atol = 1e-13
  end

  
  test_combos=[
    (make_Heisenberg_AutoMPO,"S=1/2"),
    (make_Hubbard_AutoMPO,"Electron"), 
  ]
  

  @testset "QR MPO tensors with complex block structures, H=$(test_combo[1])" for test_combo in test_combos
    N,NNN = 10,2 #10 lattice site, up 7th neight interactions
    sites = siteinds(test_combo[2], N; conserve_qns=true)
    H=test_combo[1](sites,NNN)
    for n in 5:5
      W = H[n]
      @test flux(W) == QN("Sz", 0)
      ilr = filterinds(W; tags="l=$n")[1]
      ilq = noncommoninds(W, ilr)
      Q, R, q = qr(W, ilq)
      @test flux(Q) == QN("Sz", 0) #qr should move all flux on W (0 in this case) onto R
      @test flux(R) == QN("Sz", 0) #this effectively removes all flux between Q and R in thie case.
      @test hastags(inds(R)[1],"Link,qr")
      @test hastags(inds(Q)[end],"Link,qr")
      @test W ≈ Q * R atol = 1e-13
      # blocksparse - diag is not supported so we must convert Q*Q_dagger to dense.
      # Also fails with error in permutedims so below we use norm(a-b)≈ 0.0 instead.
      # @test dense(Q*dag(prime(Q, q))) ≈ δ(Float64, q, q') atol = 1e-13
      @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

      R, Q, q = ITensors.rq(W, ilr)
      @test flux(Q) == QN("Sz", 0)
      @test flux(R) == QN("Sz", 0)
      @test W ≈ Q * R atol = 1e-13
      @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

      Q, L, q = ITensors.ql(W, ilq)
      @test flux(Q) == QN("Sz", 0)
      @test flux(L) == QN("Sz", 0)
      @test W ≈ Q * L atol = 1e-13
      @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13

      L, Q, q = ITensors.lq(W, ilr)
      @test flux(Q) == QN("Sz", 0)
      @test flux(L) == QN("Sz", 0)
      @test W ≈ Q * L atol = 1e-13
      @test norm(dense(Q * dag(prime(Q, q))) - δ(Float64, q, q')) ≈ 0.0 atol = 1e-13
    end
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
