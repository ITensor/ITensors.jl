using ITensors, Test

@testset "diagITensor (DiagBlockSparse)" begin
  @testset "diagITensor get and set elements" begin
    i = Index(QN(0) => 2, QN(1) => 3; tags="i")

    D = diagITensor(QN(), i, dag(i'))

    for b in eachnzblock(D)
      @test flux(D, b) == QN()
    end

    D[i => 1, i' => 1] = 1
    D[i => 2, i' => 2] = 2
    D[i => 3, i' => 3] = 3
    D[i => 4, i' => 4] = 4
    D[i => 5, i' => 5] = 5

    @test_throws ErrorException D[i => 1, i' => 2] = 2.0

    @test D[i => 1, i' => 1] == 1
    @test D[i => 2, i' => 2] == 2
    @test D[i => 3, i' => 3] == 3
    @test D[i => 4, i' => 4] == 4
    @test D[i => 5, i' => 5] == 5
  end

  @testset "diagITensor Tuple constructor" begin
    i = Index(QN(0) => 2, QN(1) => 3; tags="i")

    D = diagITensor((i, dag(i')))

    for b in eachnzblock(D)
      @test flux(D, b) == QN()
    end
  end

  @testset "delta" begin
    i = Index(QN(0) => 2, QN(1) => 3; tags="i")
    ĩ = sim(i; tags="i_sim")
    j = Index(QN(0) => 2, QN(1) => 3, QN(2) => 4; tags="j")

    A = randomITensor(QN(), i, dag(j))

    δiĩ = δ(dag(i), ĩ)

    @test storage(δiĩ) isa NDTensors.DiagBlockSparse{ElT,ElT} where {ElT<:Number}

    B = A * δiĩ

    A = permute(A, i, j)
    B = permute(B, ĩ, j)

    @test norm(dense(NDTensors.tensor(A)) - dense(NDTensors.tensor(B))) ≈ 0
  end

  @testset "delta Tuple constructor" begin
    i = Index(QN(0) => 2, QN(1) => 3; tags="i")
    ĩ = sim(i; tags="i_sim")

    δiĩ = δ((dag(i), ĩ))

    for b in eachnzblock(δiĩ)
      @test flux(δiĩ, b) == QN()
    end
  end

  @testset "denseblocks: convert DiagBlockSparse to BlockSparse" begin
    i = Index([QN(0) => 2, QN(1) => 3])
    A = diagITensor(i', dag(i))
    randn!(ITensors.data(A))
    B = denseblocks(A)
    for n in 1:dim(i)
      @test A[n, n] == B[n, n]
    end
    @test dense(A) == dense(B)
  end

  @testset "Regression test for QN delta contraction bug" begin
    # http://itensor.org/support/2814/block-sparse-itensor-wrong-results-multiplying-delta-tensor
    s = Index([QN(("N", i, 1)) => 1 for i in 1:2])
    l = dag(addtags(s, "left"))
    r = addtags(s, "right")
    u = addtags(s, "up")
    d = dag(addtags(s, "down"))
    A = emptyITensor(l, r, u, d)
    A[1, 1, 1, 1] = 1.0
    A[1, 1, 2, 2] = 1.0
    A[2, 2, 1, 1] = 1.0
    A[2, 2, 2, 2] = 1.0
    δlr = δ(dag(l), dag(r))
    δud = δ(dag(u), dag(d))
    A1 = A * δlr
    denseA1 = dense(A) * dense(δlr)
    A2 = A1 * δud
    denseA2 = denseA1 * dense(δud)
    @test dense(A1) ≈ denseA1
    @test dense(A2) ≈ denseA2
    @test A2[] ≈ 4
  end

  @testset "Regression test for QN delta dag, contract, and norm" begin
    i = Index([QN("Sz", 0) => 1, QN("Sz", 1) => 1])
    x = δ(i, dag(i)')

    @test isone(x[1, 1])
    @test isone(dag(x)[1, 1])

    c = 2 + 3im
    x *= c

    @test x[1, 1] == c
    @test dag(x)[1, 1] == conj(c)
    @test (x * dag(x))[] == 2 * abs2(c)
    @test (x * dag(x))[] ≈ norm(x)^2
  end

  @testset "Regression test for printing a QN Diag ITensor" begin
    # https://github.com/ITensor/NDTensors.jl/issues/61
    i = Index([QN() => 2])
    A = randomITensor(i', dag(i))
    U, S, V = svd(A, i')
    # Test printing S
    io = IOBuffer()
    show(io, S)
    sS = String(take!(io))
    @test sS isa String
    # Test printing U
    io = IOBuffer()
    show(io, U)
    sU = String(take!(io))
    @test sU isa String
  end
end

nothing
