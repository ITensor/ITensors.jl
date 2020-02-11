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

  @testset "Combine and uncombine" begin

    @testset "Order 2" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      A = randomITensor(QN(),i1,dag(i1'))

      iss = [i1,
             dag(i1'),
             (i1,dag(i1')),
             (dag(i1'),i1)]

      for is in iss
        C,c = combiner(is; tags="c")
        AC = A*C
        @test nnz(AC) == nnz(A)
        Ap = AC*dag(C)
        @test nnz(Ap) == nnz(A)
        @test nnzblocks(Ap) == nnzblocks(A)
        @test norm(A-Ap) ≈ 0.0
      end
    end

    @testset "Order 3, Combine 2" begin
      i = Index([QN(0)=>2,QN(1)=>2],"i")

      A = randomITensor(QN(0),i,dag(i)',dag(i)'')

      C,c = combiner(i,dag(i)'')

      AC = A*C

      @test hasinds(AC,c,i')
      @test nnz(AC) == nnz(A)

      for b in nzblocks(AC)
        @test flux(AC,b) == QN(0)
      end

      B = ITensor(QN(0),i',c)
      @test nnz(B) == nnz(AC)
      @test nnzblocks(B) == nnzblocks(AC)

      Ablock_221 = vec(permutedims(blockview(tensor(A),(2,2,1)),(1,3,2)))
      ACblock_32 = vec(blockview(tensor(AC),(3,2)))
      @test Ablock_221 == ACblock_32

      Ablock_111 = vec(permutedims(blockview(tensor(A),(1,1,1)),(1,3,2)))
      ACblock_21_1 = vec(blockview(tensor(AC),(2,1)))[1:length(Ablock_111)]
      @test Ablock_111 == ACblock_21_1

      Ablock_212 = vec(permutedims(blockview(tensor(A),(2,1,2)),(1,3,2)))
      ACblock_21_2 = vec(blockview(tensor(AC),(2,1)))[length(Ablock_111)+1:end]
      @test Ablock_212 == ACblock_21_2

      Ap = AC*dag(C)

      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
      @test hassameinds(A,Ap)
    end

    @testset "Order 3" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      i2 = settags(i1,"i2")
      A = randomITensor(QN(),i1,i2,dag(i1'))

      iss = [i1,
             i2,
             dag(i1'),
             (i1,i2),
             (i2,i1), 
             (i1,dag(i1')),
             (dag(i1'),i1),
             (i2,dag(i1')),
             (dag(i1'),i2),
             (i1,i2,dag(i1')),
             (i1,dag(i1'),i2), 
             (i2,i1,dag(i1')), 
             (i2,dag(i1'),i1), 
             (dag(i1'),i1,i2), 
             (dag(i1'),i2,i1)]

      for is in iss
        C,c = combiner(is; tags="c")
        AC = A*C
        @assert nnz(AC) == nnz(A)
        Ap = AC*dag(C)
        @assert nnz(Ap) == nnz(A)
        @assert nnzblocks(Ap) == nnzblocks(A)
        @assert norm(A-AC*dag(C)) ≈ 0.0
      end
    end

    @testset "Order 4" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      i2 = settags(i1,"i2")
      A = randomITensor(QN(),i1,i2,dag(i1'),dag(i2'))

      iss = [i1,
             i2,
             dag(i1'),
             dag(i2'),
             (i1,i2),
             (i2,i1),
             (i1,dag(i1')),
             (dag(i1'),i1),
             (i1,dag(i2')),
             (dag(i2'),i1),
             (i2,dag(i1')),
             (dag(i1'),i2),
             (i2,dag(i2')),
             (dag(i2'),i2),
             (dag(i1'),dag(i2')),
             (dag(i2'),dag(i1')),
             (i1,i2,dag(i1')),
             (i1,dag(i1'),i2),
             (i2,i1,dag(i1')),
             (i2,dag(i1'),i1),
             (dag(i1'),i1,i2),
             (dag(i1'),i2,i1),
             (i1,dag(i1'),dag(i2')),
             (i1,dag(i2'),dag(i1')),
             (dag(i1'),i1,dag(i2')),
             (dag(i1'),dag(i2'),i1),
             (dag(i2'),i1,dag(i1')),
             (dag(i2'),dag(i1'),i1),
             (i1,i2,dag(i1'),dag(i2')),
             (i1,i2,dag(i2'),dag(i1')),
             (i1,dag(i1'),i2,dag(i2')),
             (i1,dag(i1'),dag(i2'),i2),
             (i1,dag(i2'),i2,dag(i1')),
             (i1,dag(i2'),dag(i1'),i2),
             (i2,i1,dag(i1'),dag(i2')),
             (i2,i1,dag(i2'),dag(i1')),
             (i2,dag(i1'),i1,dag(i2')),
             (i2,dag(i1'),dag(i2'),i1),
             (i2,dag(i2'),i1,dag(i1')),
             (i2,dag(i2'),dag(i1'),i1),
             (dag(i1'),i2,i1,dag(i2')),
             (dag(i1'),i2,dag(i2'),i1),
             (dag(i1'),i1,i2,dag(i2')),
             (dag(i1'),i1,dag(i2'),i2),
             (dag(i1'),dag(i2'),i2,i1),
             (dag(i1'),dag(i2'),i1,i2),
             (dag(i2'),i1,dag(i1'),i2),
             (dag(i2'),i1,i2,dag(i1')),
             (dag(i2'),dag(i1'),i1,i2),
             (dag(i2'),dag(i1'),i2,i1),
             (dag(i2'),i2,i1,dag(i1')),
             (dag(i2'),i2,dag(i1'),i1)]

      for is in iss
        C,c = combiner(is; tags="c")
        AC = A*C
        @test nnz(AC) == nnz(A)
        Ap = AC*dag(C)
        @test nnz(Ap) == nnz(A)
        @test nnzblocks(Ap) == nnzblocks(A)
        @test norm(A-Ap) ≈ 0.0
      end
    end

    @testset "Order 4, Combine 2, Example 1" begin
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
      @test nnz(B) == nnz(AC)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)
      
      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

    @testset "Order 4, Combine 2, Example 2" begin
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

      B = ITensor(QN(),s1,s2,c)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)

      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

    @testset "Order 4, Combine 2, Example 3" begin
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

      B = ITensor(QN(),s1,s2,c)
      @test nnzblocks(B) == nnzblocks(AC)

      Ap = AC*dag(C)

      @test hassameinds(A,Ap)
      @test norm(A-Ap) == 0
      @test nnz(A) == nnz(Ap)
      @test nnzblocks(A) == nnzblocks(Ap)
    end

  @testset "Contract to scalar" begin
    i = Index([QN(0)=>1,QN(1)=>1],"i")
    A = randomITensor(QN(0),i,dag(i'))

    c = A*dag(A)

    @test nnz(c) == 1
    @test nnzblocks(c) == 1
    @test c[] isa Float64
    @test c[] ≈ norm(A)^2
  end

end

