using ITensors,
      Test

@testset "diagITensor (DiagBlockSparse)" begin

  @testset "diagITensor get and set elements" begin
    i = Index(QN(0)=>2,QN(1)=>3; tags="i")

    D = diagITensor(QN(),i,dag(i'))

    for b in eachnzblock(D)
      @test flux(D,b) == QN()
    end

    D[i(1),i'(1)] = 1
    D[i(2),i'(2)] = 2
    D[i(3),i'(3)] = 3
    D[i(4),i'(4)] = 4
    D[i(5),i'(5)] = 5

    @test_throws ErrorException D[i(1),i'(2)] = 2.0

    @test D[i(1),i'(1)] == 1
    @test D[i(2),i'(2)] == 2
    @test D[i(3),i'(3)] == 3
    @test D[i(4),i'(4)] == 4
    @test D[i(5),i'(5)] == 5
  end

  @testset "diagITensor Tuple constructor" begin
    i = Index(QN(0)=>2, QN(1)=>3; tags="i")

    D = diagITensor((i, dag(i')))

    for b in eachnzblock(D)
      @test flux(D, b) == QN()
    end
  end

  @testset "delta" begin
    i = Index(QN(0)=>2,QN(1)=>3; tags="i")
    ĩ = sim(i; tags="i_sim")
    j = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="j")

    A = randomITensor(QN(), i, dag(j))

    δiĩ = δ(dag(i), ĩ)

    @test store(δiĩ) isa NDTensors.DiagBlockSparse{ElT,
                                                   ElT} where {ElT<:Number}

    B = A * δiĩ

    A = permute(A, i, j)
    B = permute(B, ĩ, j)

    @test norm(dense(NDTensors.tensor(A)) -
               dense(NDTensors.tensor(B))) ≈ 0
  end

  @testset "delta Tuple constructor" begin
    i = Index(QN(0)=>2, QN(1)=>3; tags="i")
    ĩ = sim(i; tags="i_sim")

    δiĩ = δ((dag(i), ĩ))

    for b in eachnzblock(δiĩ)
      @test flux(δiĩ, b) == QN()
    end
  end

end

nothing
