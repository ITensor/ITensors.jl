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

  @testset "Random constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = randomITensor(QN(1),i,dag(j))

    @test flux(A) == QN(1)
    @test nnzblocks(A) == 1
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

  @testset "Combine and uncombine" begin

    @testset "Example 1" begin
      i = Index([QN(0)=>2,QN(1)=>2],"i")

      A = randomITensor(QN(0),i,dag(i)',dag(i)'')

      C,c = combiner(i,dag(i)'')

      AC = A*C

      @test hasinds(AC,c,i')
      @test nnz(AC) == nnz(A)

      for b in nzblocks(AC)
        @test flux(AC,b) == QN(0)
      end

      @show nzblocks(AC)

      # Check (2,2,1) and (2,3) are the same data
      @test reshape(permutedims(blockview(tensor(A),(2,2,1)),(1,3,2)),4,2) == blockview(tensor(AC),(3,2))

      # Check (1,1,1) and the beginning of block (1,2) are the same data
      @test reshape(permutedims(blockview(tensor(A),(1,1,1)),(1,3,2)),4,2) == blockview(tensor(AC),(2,1))[1:4,1:2]

      # Check (2,1,2) and the end of block (1,2) are the same data
      @test reshape(permutedims(blockview(tensor(A),(2,1,2)),(1,3,2)),4,2) == blockview(tensor(AC),(2,1))[5:8,1:2]

      Ap = AC*dag(C)

      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
      @test hassameinds(A,Ap)
    end

    @testset "Example 2" begin
      s1 = Index([QN(("Sz", 0),("Nf",0))=>1,
                  QN(("Sz",+1),("Nf",1))=>1,
                  QN(("Sz",-1),("Nf",1))=>1,
                  QN(("Sz", 0),("Nf",2))=>1],"site,n=1");
      s2 = replacetags(s1,"n=1","n=2")

      A = randomITensor(QN(),s1,s2,dag(s1)',dag(s2)')

      C,c = combiner(dag(s1)',dag(s2)')

      AC = A*C

      @test norm(AC) ≈ norm(A)
      @test hasinds(AC,s1,s2,c)
      @test nnz(AC) == nnz(A)
      for b in nzblocks(AC)
        @test flux(AC,b) == QN()
      end

      @test nnzblocks(AC) < nnz(A)

      B = ITensor(QN(),s1,s2,c)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)
      
      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

    @testset "Example 3" begin
      s1 = Index([QN(("Nf",0))=>1,
                  QN(("Nf",1))=>1],"site,n=1")
      s2 = replacetags(s1,"n=1","n=2")

      A = randomITensor(QN(),dag(s2)',s2,dag(s1)',s1)

      C,c = combiner(dag(s2)',dag(s1)')

      AC = A*C

      @test norm(AC) ≈ norm(A)
      @test hasinds(AC,s1,s2,c)
      @test nnz(AC) == nnz(A)
      for b in nzblocks(AC)
        @test flux(AC,b) == QN()
      end

      @test blockview(tensor(A),(1,1,1,1))[1] == blockview(tensor(AC),(1,1,1))[1]
      @test blockview(tensor(A),(2,2,1,1))[1] == blockview(tensor(AC),(2,1,2))[1]
      @test blockview(tensor(A),(1,2,2,1))[1] == blockview(tensor(AC),(2,1,2))[2]
      @test blockview(tensor(A),(2,1,1,2))[1] == blockview(tensor(AC),(1,2,2))[1]
      @test blockview(tensor(A),(1,1,2,2))[1] == blockview(tensor(AC),(1,2,2))[2]
      @test blockview(tensor(A),(2,2,2,2))[1] == blockview(tensor(AC),(2,2,3))[1]

      B = ITensor(QN(),s1,s2,c)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)

      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

    @testset "Example 4" begin
      s1 = Index([QN(("Nf",0))=>1,
                  QN(("Nf",1))=>1],"site,n=1")
      s2 = replacetags(s1,"n=1","n=2")

      A = randomITensor(QN(),dag(s1)',s2,dag(s2)',s1)

      C,c = combiner(dag(s2)',dag(s1)')

      AC = A*C

      @test norm(AC) ≈ norm(A)
      @test hasinds(AC,s1,s2,c)
      @test nnz(AC) == nnz(A)
      for b in nzblocks(AC)
        @test flux(AC,b) == QN()
      end

      @test blockview(tensor(A),(1,1,1,1))[1] == blockview(tensor(AC),(1,1,1))[1]
      @test blockview(tensor(A),(2,2,1,1))[1] == blockview(tensor(AC),(2,1,2))[1]
      @test blockview(tensor(A),(1,2,2,1))[1] == blockview(tensor(AC),(2,1,2))[2]
      @test blockview(tensor(A),(2,1,1,2))[1] == blockview(tensor(AC),(1,2,2))[1]
      @test blockview(tensor(A),(1,1,2,2))[1] == blockview(tensor(AC),(1,2,2))[2]
      @test blockview(tensor(A),(2,2,2,2))[1] == blockview(tensor(AC),(2,2,3))[1]

      B = ITensor(QN(),s1,s2,c)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)

      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

  end

end

