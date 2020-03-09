using ITensors,
      Test

@testset "diagITensor (DiagBlockSparse)" begin

  @testset "diagITensor get and set elements" begin
    i = Index(QN(0)=>2,QN(1)=>3; tags="i")

    D = diagITensor(QN(),i,dag(i'))

    for n in nnzblocks(D)
      b = block(D,n)
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

  @testset "delta" begin
    i = Index(QN(0)=>2,QN(1)=>3; tags="i")
    ĩ = sim(i; tags="i_sim")
    j = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="j")

    A = randomITensor(QN(),i,dag(j))

    δiĩ = δ(dag(i),ĩ)

    @test store(δiĩ) isa DiagBlockSparse{ElT,ElT} where {ElT<:Number}

    B = A*δiĩ

    A = permute(A,i,j)
    B = permute(B,ĩ,j)

    @test norm(dense(tensor(A))-dense(tensor(B))) ≈ 0
  end

end

