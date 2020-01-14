using ITensors,
      Test

@testset "BlockSparse ITensor" begin

  @testset "Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(-1)=>4,QN(-2)=>5],"j")

    A = ITensor(QN(0),i,j)

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(-1)=>4,QN(-2)=>5],"j")

    A = randomITensor(QN(0),i,j)

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Contraction" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = randomITensor(QN(0),i,dag(j))

    @show A

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2

    B = randomITensor(QN(0),j,dag(i)')

    @show B

    @test flux(B) == QN(0)
    @test nnzblocks(B) == 2

    C = A*B

    @show C

    @test hasinds(C,i,i')
    @test flux(C) == QN(0)
    @test nnzblocks(C) == 2
  end

end

