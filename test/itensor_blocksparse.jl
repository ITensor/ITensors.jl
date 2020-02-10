using ITensors,
      Test

@testset "BlockSparse ITensor" begin

  @testset "Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = ITensor(QN(0),i,dag(j))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Empty constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")

    A = ITensor(i,dag(i'))

    @test nnzblocks(A) == 0
    @test nnz(A) == 0
    @test hasinds(A,i,i')
    @test isnothing(flux(A))

    A[i(1),i'(1)] = 1.0

    @test nnzblocks(A) == 1
    @test nnz(A) == 1
    @test flux(A) == QN(0)

    A[i(2),i'(2)] = 1.0

    @test nnzblocks(A) == 2
    @test nnz(A) == 5
    @test flux(A) == QN(0)
  end

  @testset "Random constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = randomITensor(QN(1),i,dag(j))

    @test flux(A) == QN(1)
    @test nnzblocks(A) == 1
  end

  @testset "setindex!" begin

    @testset "Test 1" begin
      s1 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s1")
      s2 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s2")
      A = ITensor(s1,s2)

      @test nnzblocks(A) == 0
      @test nnz(A) == 0
      @test hasinds(A,s1,s2)
      @test isnothing(flux(A))

      A[2,1] = 1.0/sqrt(2)

      @test nnzblocks(A) == 1
      @test nnz(A) == 1
      @test A[s1(2),s2(1)] ≈ 1.0/sqrt(2)
      @test flux(A) == QN("N",1,-1)

      A[1,2] = 1.0/sqrt(2)

      @test nnzblocks(A) == 2
      @test nnz(A) == 2
      @test A[s1(2),s2(1)] ≈ 1.0/sqrt(2)
      @test A[s1(1),s2(2)] ≈ 1.0/sqrt(2)
      @test flux(A) == QN("N",1,-1)
    end

    @testset "Test 2" begin
      s1 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s1")
      s2 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s2")
      A = ITensor(s1,s2)

      @test nnzblocks(A) == 0
      @test nnz(A) == 0
      @test hasinds(A,s1,s2)
      @test isnothing(flux(A))

      A[1,2] = 1.0/sqrt(2)

      @test nnzblocks(A) == 1
      @test nnz(A) == 1
      @test A[s1(1),s2(2)] ≈ 1.0/sqrt(2)
      @test flux(A) == QN("N",1,-1)

      A[2,1] = 1.0/sqrt(2)

      @test nnzblocks(A) == 2
      @test nnz(A) == 2
      @test A[s1(2),s2(1)] ≈ 1.0/sqrt(2)
      @test A[s1(1),s2(2)] ≈ 1.0/sqrt(2)
      @test flux(A) == QN("N",1,-1)
    end

  end

  @testset "Multiply by scalar" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4],"j")

    A = randomITensor(QN(0),i,dag(j))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2

    B = 2*A

    @test flux(B) == QN(0)
    @test nnzblocks(B) == 2

    for ii in dim(i), jj in dim(j)
      @test 2*A[i(ii),j(jj)] == B[i(ii),j(jj)]
    end
  end

  @testset "Copy" begin
    s = Index([QN(0)=>1,QN(1)=>1],"s")
    T = randomITensor(QN(0),s,s')
    cT = copy(T)
    for ss in dim(s), ssp in dim(s')
      @test T[s(ss),s'(ssp)] == cT[s(ss),s'(ssp)]
    end
  end

  @testset "Permute" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4],"j")

    A = randomITensor(QN(1),i,dag(j))

    @test flux(A) == QN(1)
    @test nnzblocks(A) == 1

    B = permute(A,j,i)

    @test flux(B) == QN(1)
    @test nnzblocks(B) == 1

    for ii in dim(i), jj in dim(j)
      @test A[ii,jj] == B[jj,ii]
    end
  end

  @testset "Contraction" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4],"j")

    A = randomITensor(QN(0),i,dag(j))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2

    B = randomITensor(QN(1),j,dag(i)')

    @test flux(B) == QN(1)
    @test nnzblocks(B) == 1

    C = A*B

    @test hasinds(C,i,i')
    @test flux(C) == QN(1)
    @test nnzblocks(C) == 1
  end

end

