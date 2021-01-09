using ITensors,
      Test,
      Random

Random.seed!(1234)

@testset "BlockSparse ITensor" begin

  @testset "Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = ITensor(QN(0),i,dag(j))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Construct from Array" begin
    i = Index([QN(0)=>1, QN(1)=>2], "i")

    A = [1.0 0.0 0.0;
         0.0 2.0 3.0;
         0.0 1e-10 4.0]
    T = ITensor(A, i', dag(i))
    @test flux(T) == QN(0)
    @test nnzblocks(T) == 2
    @test Block(1,1) in nzblocks(T)
    @test Block(2,2) in nzblocks(T)
    @test T[1, 1] == 1.0
    @test T[2, 2] == 2.0
    @test T[2, 3] == 3.0
    @test T[3, 2] == 1e-10
    @test T[3, 3] == 4.0

    T = itensor(A, i', dag(i))
    @test flux(T) == QN(0)
    @test nnzblocks(T) == 2
    @test Block(1,1) in nzblocks(T)
    @test Block(2,2) in nzblocks(T)
    @test T[1, 1] == 1.0
    @test T[2, 2] == 2.0
    @test T[2, 3] == 3.0
    @test T[3, 2] == 1e-10
    @test T[3, 3] == 4.0

    T = ITensor(A, i', dag(i); tol = 1e-9)
    @test flux(T) == QN(0)
    @test nnzblocks(T) == 2
    @test Block(1,1) in nzblocks(T)
    @test Block(2,2) in nzblocks(T)
    @test T[1, 1] == 1.0
    @test T[2, 2] == 2.0
    @test T[2, 3] == 3.0
    @test T[3, 2] == 0.0
    @test T[3, 3] == 4.0

    A = [1e-9 0.0 0.0;
         0.0 2.0 3.0;
         0.0 1e-10 4.0]
    T = ITensor(A, i', dag(i); tol = 1e-8)
    @test flux(T) == QN(0)
    @test nnzblocks(T) == 1
    @test Block(2,2) in nzblocks(T)
    @test T[1, 1] == 0.0
    @test T[2, 2] == 2.0
    @test T[2, 3] == 3.0
    @test T[3, 2] == 0.0
    @test T[3, 3] == 4.0

    A = [1e-9 2.0 3.0;
         1e-9 1e-10 2e-10;
         2e-9 1e-10 4e-10]
    T = ITensor(A, i', dag(i); tol = 1e-8)
    @test flux(T) == QN(-1)
    @test nnzblocks(T) == 1
    @test Block(1,2) in nzblocks(T)
    @test T[1, 1] == 0.0
    @test T[1, 2] == 2.0
    @test T[1, 3] == 3.0
    @test T[2, 2] == 0.0
    @test T[2, 3] == 0.0
    @test T[3, 2] == 0.0
    @test T[3, 3] == 0.0

    A = [1e-9 2.0 3.0;
         1e-5 1e-10 2e-10;
         2e-9 1e-10 4e-10]
    @test_throws ErrorException ITensor(A, i', dag(i); tol = 1e-8)
  end

  @testset "ITensor iteration" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = randomITensor(i, dag(j))
    Is = eachindex(A)
    @test length(Is) == dim(A)
    sumA = 0.0
    for I in Is
      sumA += A[I]
    end
    @test sumA ≈ sum(ITensors.data(A))
    sumA = 0.0
    for a in A
      sumA += a
    end
    @test sumA ≈ sum(A)
  end

  @testset "Constructor (from Tuple)" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = ITensor(QN(0), (i, dag(j)))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Constructor (no flux specified)" begin
    i = Index([QN(0)=>1, QN(1)=>2], "i")
    j = Index([QN(0)=>3, QN(1)=>4, QN(2)=>5], "j")

    A = ITensor(i, dag(j))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "Constructor (Tuple, no flux specified)" begin
    i = Index([QN(0)=>1, QN(1)=>2], "i")
    j = Index([QN(0)=>3, QN(1)=>4, QN(2)=>5], "j")

    A = ITensor((i, dag(j)))

    @test flux(A) == QN(0)
    @test nnzblocks(A) == 2
  end

  @testset "No indices getindex" begin
    T = ITensor(QN())
    @test order(T) == 0
    @test flux(T) == nothing
    @test nnzblocks(T) == 1
    @test T[] == 0

    s = Index(QN(-1)=>1,QN(1)=>1)
    A = emptyITensor(s, dag(s'))
    B = emptyITensor(s', dag(s))
    A[1, 1] = 1
    B[2, 2] = 1
    C = A * B
    @test order(C) == 0
    @test flux(C) == nothing
    @test nnzblocks(C) == 0
    @test C[] == 0
  end

  @testset "Empty constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")

    A = emptyITensor(i,dag(i'))

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

  @testset "Check flux when setting elements" begin
    i = Index(QN(0)=>1,QN(1)=>1; tags="i")
    A = randomITensor(QN(0),i,dag(i'))
    @test_throws ErrorException A[i(1),i'(2)] = 1.0
  end

  @testset "Random constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    A = randomITensor(QN(1),i,dag(j))

    @test flux(A) == QN(1)
    @test nnzblocks(A) == 1
    
    B = randomITensor(i,dag(j))

    @test flux(B) == QN()
    @test nnzblocks(B) == 2
  end


  @testset "QN setelt" begin
    i = Index(QN(0)=>2,QN(1)=>2,tags="i")

    T = setelt(i(1))
    @test T[i(1)] ≈ 1.0
    @test T[i(2)] ≈ 0.0
    @test T[i(3)] ≈ 0.0
    @test T[i(4)] ≈ 0.0

    T = setelt(i(2))
    @test T[i(1)] ≈ 0.0
    @test T[i(2)] ≈ 1.0
    @test T[i(3)] ≈ 0.0
    @test T[i(4)] ≈ 0.0

    # Test setelt taking Pair{Index,Int}
    T = setelt(i=>3)
    @test T[i(1)] ≈ 0.0
    @test T[i(2)] ≈ 0.0
    @test T[i(3)] ≈ 1.0
    @test T[i(4)] ≈ 0.0
  end


  @testset "setindex!" begin

    @testset "Test 1" begin
      s1 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s1")
      s2 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s2")
      A = emptyITensor(s1,s2)

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
      A = emptyITensor(s1,s2)

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
      @test 2*A[i=>ii,j=>jj] == B[i=>ii,j=>jj]
    end
  end

  @testset "Check arrows when summing" begin
    s = siteinds("S=1/2",4;conserve_qns=true)
    Tout = randomITensor(QN("Sz"=>2),s[2],s[1],s[3],s[4])
    Tin = randomITensor(QN("Sz"=>2),dag(s[1]),dag(s[2]),dag(s[3]),dag(s[4]))
    @test norm(Tout-Tout) < 1E-10 # this is ok
    @test_throws ErrorException (Tout+Tin)         # not ok
  end

  @testset "Copy" begin
    s = Index([QN(0)=>1,QN(1)=>1],"s")
    T = randomITensor(QN(0),s,s')
    cT = copy(T)
    for ss in dim(s), ssp in dim(s')
      @test T[s=>ss,s'=>ssp] == cT[s=>ss,s'=>ssp]
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

    @testset "Combine no indices" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      A = randomITensor(QN(),i1,dag(i1'))

      C = combiner()
      c = combinedind(C)
      @test isnothing(c)
      AC = A*C
      @test nnz(AC) == nnz(A)
      @test nnzblocks(AC) == nnzblocks(A)
      @test hassameinds(AC,A)
      @test norm(AC-A*C) ≈ 0.0
      Ap = AC*dag(C)
      @test nnz(Ap) == nnz(A)
      @test nnzblocks(Ap) == nnzblocks(A)
      @test hassameinds(Ap,A)
      @test norm(A-Ap) ≈ 0.0
    end

    @testset "Combine set direction" begin
      i1 = Index([QN(0)=>2,QN(1)=>3],"i1")
      A = randomITensor(i1',dag(i1))
      @test isnothing(ITensors.checkflux(A))
      C = combiner(dag(i1); dir = ITensors.Out)
      c = combinedind(C)
      @test dir(c) == ITensors.Out
      AC = A * C
      @test nnz(AC) == nnz(A)
      @test nnzblocks(AC) == nnzblocks(A)
      @test isnothing(ITensors.checkflux(AC))
      Ap = AC*dag(C)
      @test nnz(Ap) == nnz(A)
      @test nnzblocks(Ap) == nnzblocks(A)
      @test hassameinds(Ap, A)
      @test isnothing(ITensors.checkflux(AC))
      @test A ≈ Ap
    end

    @testset "Order 2 (IndexSet constructor)" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      A = randomITensor(QN(),i1,dag(i1'))

      iss = [i1,
             dag(i1'),
             (i1,dag(i1')),
             (dag(i1'),i1)]

      for is in iss
        C = combiner(IndexSet(is); tags="c")
        AC = A*C
        @test nnz(AC) == nnz(A)
        Ap = AC*dag(C)
        @test nnz(Ap) == nnz(A)
        @test nnzblocks(Ap) == nnzblocks(A)
        @test norm(A-Ap) ≈ 0.0
      end
    end

    @testset "Order 2" begin
      i1 = Index([QN(0,2)=>2,QN(1,2)=>2],"i1")
      A = randomITensor(QN(), (i1, dag(i1')))

      iss = [i1,
             dag(i1'),
             (i1,dag(i1')),
             (dag(i1'),i1)]

      for is in iss
        C = combiner(is; tags="c")
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

      C = combiner(i,dag(i)'')
      c = combinedind(C)

      AC = A*C

      @test hasinds(AC,c,i')
      @test nnz(AC) == nnz(A)

      for b in nzblocks(AC)
        @test flux(AC,b) == QN(0)
      end

      B = ITensor(QN(0),i',c)
      @test nnz(B) == nnz(AC)
      @test nnzblocks(B) == nnzblocks(AC)

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
        C = combiner(is; tags="c")
        AC = A*C
        @test nnz(AC) == nnz(A)
        Ap = AC*dag(C)
        @test nnz(Ap) == nnz(A)
        @test nnzblocks(Ap) == nnzblocks(A)
        @test norm(A-AC*dag(C)) ≈ 0.0
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
        C = combiner(is; tags="c")
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

      C = combiner(dag(s1)',dag(s2)')
      c = combinedind(C)

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

      C = combiner(dag(s2)',dag(s1)')
      c = combinedind(C)

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

      C = combiner(dag(s2)',dag(s1)')
      c = combinedind(C)

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
  end

  @testset "Check that combiner commutes" begin
    i = Index(QN(0,2)=>2,QN(1,2)=>2; tags="i")
    j = settags(i,"j")
    A = randomITensor(QN(0,2),i,j,dag(i'),dag(j'))
    C = combiner(i,j)
    @test norm(A*dag(C')*C-A*C*dag(C')) ≈ 0.0
  end

  @testset "Combiner for block deficient ITensor" begin
    i = Index(QN(0,2)=>2,QN(1,2)=>2; tags="i")
    j = settags(i,"j")
    A = emptyITensor(i,j,dag(i'))
    A[1,1,1] = 1.0
    C = combiner(i,j; tags="c")
    AC = A*C
    Ap = AC*dag(C)
    @test norm(A-Ap) ≈ 0.0
    @test norm(Ap-A) ≈ 0.0
  end

  @testset "Combine Complex ITensor" begin
    s1 = Index(QN(("Sz",1)) => 1,QN(("Sz",-1)) => 1;tags="S=1/2,Site,n=1")
    s2 = Index(QN(("Sz",1)) => 1,QN(("Sz",-1)) => 1;tags="S=1/2,Site,n=2")

    T = randomITensor(ComplexF64,QN("Sz",0),s1,s2)

    C = combiner(s1,s2)
    CT = C*T
    @test norm(CT) ≈ norm(T)
    TT = dag(C)*CT
    @test TT ≈ T
  end

  @testset "Combiner bug #395" begin
    i1 = Index([QN(0)=>1,QN(1)=>2],"i1")
    i2 = Index([QN(0)=>1,QN(1)=>2],"i2")
    A = randomITensor(QN(),i1,i2,dag(i1)', dag(i2)')
    CL = combiner(i1,i2)
    CR = combiner(dag(i1)',dag(i2)')
    AC = A * CR * CL
    @test AC * dag(CR) * dag(CL) ≈ A
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

  @testset "eigen" begin

    @testset "eigen hermitian" begin
      i = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="i")
      j = settags(i,"j")
      k = settags(i,"k")
      l = settags(i,"l")

      A = randomITensor(QN(),i,j,dag(k),dag(l))
      A = A*prime(dag(A),(i,j))

      F = eigen(A; ishermitian=true, tags="x")

      D, U = F
      Ut = F.Vt


      @test store(U) isa NDTensors.BlockSparse
      @test store(D) isa NDTensors.DiagBlockSparse

      u = commonind(D,U)
      up = uniqueind(D,U)

      @test hastags(u,"x")
      @test plev(u) == 0
      @test hastags(up,"x")
      @test plev(up) == 1

      @test hassameinds(U, (i,j,u))
      @test hassameinds(D, (u,up))

      @test A ≈ dag(U) * D * U' atol=1e-11
      @test A ≈ dag(U) * D * Ut atol=1e-11
      @test A * U ≈ U' * D atol=1e-11
      @test A * U ≈ Ut * D atol=1e-11
    end

    @testset "eigen hermitian (truncate)" begin
      i = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="i")
      j = settags(i,"j")
      k = settags(i,"k")
      l = settags(i,"l")

      A = randomITensor(QN(),i,j,dag(k),dag(l))
      A = A*prime(dag(A),(i,j))
      for i = 1:4
        A = mapprime(A*A',2,1)
      end
      A = A/norm(A)

      cutoff = 1e-5
      F = eigen(A; ishermitian=true,
                   tags="x",
                   cutoff=cutoff)

      D, U, spec = F
      Ut = F.Vt

      @test store(U) isa NDTensors.BlockSparse
      @test store(D) isa NDTensors.DiagBlockSparse

      u = commonind(D, U)
      up = uniqueind(D, U)

      @test hastags(u,"x")
      @test plev(u) == 0
      @test hastags(up,"x")
      @test plev(up) == 1

      @test hassameinds(U,(i,j,u))
      @test hassameinds(D,(u,up))

      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(D)
        @test flux(D,b)==QN(0)
      end

      Ap = dag(U) * D * U'

      @test norm(Ap-A) ≤ 1e-2
      @test norm(dag(U) * D * Ut - A) ≤ 1e-2
      @test minimum(dims(D)) == length(spec.eigs)
      @test minimum(dims(D)) < dim(i)*dim(j)

      @test spec.truncerr ≤ cutoff
      err = sqrt(1-(Ap*dag(Ap))[]/(A*dag(A))[])
      @test err ≤ cutoff
      @test err ≈ spec.truncerr rtol=3e-1
		end

    @testset "eigen non-hermitian" begin
      i = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="i")
      j = settags(i,"j")

      A = randomITensor(QN(),i,j,dag(i'),dag(j'))

      F = eigen(A; tags="x")

      D, U = F
      Ut = F.Vt

      @test store(U) isa NDTensors.BlockSparse
      @test store(D) isa NDTensors.DiagBlockSparse

      u = commonind(D,U)
      up = uniqueind(D,U)

      @test hastags(u,"x")
      @test plev(u) == 0
      @test hastags(up,"x")
      @test plev(up) == 1

      @test A ≉ U' * D * dag(U) atol=1e-12
      @test A ≉ Ut * D * dag(U) atol=1e-12
      @test A * U ≈ U' * D atol=1e-12
      @test A * U ≈ Ut * D atol=1e-12
    end

    @testset "eigen non-hermitian (general inds)" begin
      i = Index(QN(0) => 2,
                QN(1) => 3,
                QN(2) => 4; tags="i")
      j = settags(i, "j")
      ĩ, j̃ = sim(i), sim(j)

      A = randomITensor(QN(), i, j, dag(ĩ),dag(j̃))

      F = eigen(A, (i, j), (ĩ, j̃); lefttags = "x", righttags = "y")

      D, U = F
      Ut = F.Vt

      @test store(U) isa NDTensors.BlockSparse
      @test store(D) isa NDTensors.DiagBlockSparse

      l = uniqueind(D, U)
      r = commonind(D, U)

      @test F.l == l
      @test F.r == r

      @test hastags(l, "x")
      @test plev(l) == 0
      @test hastags(r, "y")
      @test plev(r) == 0

      @test hassameinds(U, (ĩ, j̃, r))
      @test hassameinds(Ut, (i, j, l))

      @test A * U ≈ Ut * D atol=1e-12
      @test A ≉ Ut * D * dag(U) atol=1e-12
    end

    @testset "eigen mixed arrows" begin
      i1 = Index([QN(0)=>1,QN(1)=>2],"i1")
      i2 = Index([QN(0)=>1,QN(1)=>2],"i2")
      A = randomITensor(i1, i2, dag(i1)', dag(i2)')
      F = eigen(A, (i1, i1'), (i2', i2))
      D, U = F
      Ut = F.Vt
      @test A * U ≈ Ut * D atol=1e-12
    end

  end

  @testset "svd" for ElT ∈ (Float64, ComplexF64)

    @testset "svd example 1" begin
      i = Index(QN(0)=>2,QN(1)=>2; tags="i")
      j = Index(QN(0)=>2,QN(1)=>2; tags="j")
      A = randomITensor(ElT, QN(0),i,dag(j))
      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      U,S,V = svd(A,i)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd example 2" begin
      i = Index(QN(0)=>5,QN(1)=>6; tags="i")
      j = Index(QN(-1)=>2,QN(0)=>3,QN(1)=>4; tags="j")
      A = randomITensor(ElT, QN(0),i,j)
      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      U,S,V = svd(A,i)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd example 3" begin
      i = Index(QN(0)=>5,QN(1)=>6; tags="i")
      j = Index(QN(-1)=>2,QN(0)=>3,QN(1)=>4; tags="j")
      A = randomITensor(ElT, QN(0),i,dag(j))
      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      U,S,V = svd(A,i)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd example 4" begin
			i = Index(QN(0,2)=>2,QN(1,2)=>2; tags="i")
			j = settags(i,"j")

			A = randomITensor(ElT, QN(0,2),i,j,dag(i'),dag(j'))

			U,S,V = svd(A,i,j)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(A)
        @test flux(A,b)==QN(0,2)
      end
      U,S,V = svd(A,i)
      for b in nzblocks(U)
        @test flux(U,b)==QN(0,2)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0,2)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0,2)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd example 5" begin
			i = Index(QN(0,2)=>2,QN(1,2)=>2; tags="i")
			j = settags(i,"j")

			A = randomITensor(ElT, QN(1,2),i,j,dag(i'),dag(j'))

			U,S,V = svd(A,i,j)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(A)
        @test flux(A,b)==QN(1,2)
      end
      U,S,V = svd(A,i)
      for b in nzblocks(U)
        @test flux(U,b)==QN(0,2)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(1,2)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0,2)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd example 6" begin
			i = Index(QN(0,2)=>2,QN(1,2)=>2; tags="i")
			j = settags(i,"j")

			A = randomITensor(ElT, QN(1,2),i,j,dag(i'),dag(j'))

			U,S,V = svd(A,i,i')

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      for b in nzblocks(A)
        @test flux(A,b)==QN(1,2)
      end
      U,S,V = svd(A,i)
      for b in nzblocks(U)
        @test flux(U,b)==QN(0,2)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(1,2)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0,2)
      end
      @test U * S * V ≈ A atol=1e-14
    end

    @testset "svd truncation example 1" begin
      i = Index(QN(0)=>2,QN(1)=>3; tags="i")
      j = settags(i,"j")
      A = randomITensor(ElT, QN(0),i,j,dag(i'),dag(j'))
      for i = 1:4
        A = mapprime(A*A',2,1)
      end
      A = A/norm(A)

      cutoff = 1e-5
      U,S,V,spec = svd(A,i,j; utags="x", vtags="y", cutoff=cutoff)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      u = commonind(S,U)
      v = commonind(S,V)

      @test hastags(u,"x")
      @test hastags(v,"y")

      @test hassameinds(U,(i,j,u))
      @test hassameinds(V,(i',j',v))

      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end

      Ap = U*S*V

      @test norm(Ap-A) ≤ 1e-2
      @test minimum(dims(S)) == length(spec.eigs)
      @test minimum(dims(S)) < dim(i)*dim(j)

      @test spec.truncerr ≤ cutoff
      err = real(1-(Ap*dag(Ap))[]/(A*dag(A))[])
      @test err ≤ cutoff
      @test isapprox(err,spec.truncerr; rtol=1e-6)
    end

    @testset "svd truncation example 2" begin
      i = Index(QN(0)=>3,QN(1)=>2; tags="i")
      j = settags(i,"j")
      A = randomITensor(ElT, QN(0),i,j,dag(i'),dag(j'))

      maxdim = 4
      U,S,V,spec = svd(A,i,j; utags="x", vtags="y", maxdim=maxdim)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      u = commonind(S,U)
      v = commonind(S,V)

      @test hastags(u,"x")
      @test hastags(v,"y")

      @test hassameinds(U,(i,j,u))
      @test hassameinds(V,(i',j',v))

      for b in nzblocks(A)
        @test flux(A,b)==QN(0)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(0)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end

      @test minimum(dims(S)) == maxdim
      @test minimum(dims(S)) == length(spec.eigs)
      @test minimum(dims(S)) < dim(i)*dim(j)

      Ap = U*S*V
      err = 1-(Ap*dag(Ap))[]/(A*dag(A))[]
      @test isapprox(err,spec.truncerr; rtol=1e-6)
    end

    @testset "svd truncation example 3" begin
      i = Index(QN(0)=>2,QN(1)=>3,QN(2)=>4; tags="i")
      j = settags(i,"j")
      A = randomITensor(ElT, QN(1),i,j,dag(i'),dag(j'))

      maxdim = 4
      U,S,V,spec = svd(A,i,j; utags="x", vtags="y", maxdim=maxdim)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      u = commonind(S,U)
      v = commonind(S,V)

      @test hastags(u,"x")
      @test hastags(v,"y")

      @test hassameinds(U,(i,j,u))
      @test hassameinds(V,(i',j',v))

      for b in nzblocks(A)
        @test flux(A,b)==QN(1)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(1)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0)
      end

      @test minimum(dims(S)) == maxdim
      @test minimum(dims(S)) == length(spec.eigs)
      @test minimum(dims(S)) < dim(i)*dim(j)

      Ap = U*S*V
      err = 1-(Ap*dag(Ap))[]/(A*dag(A))[]
      @test isapprox(err,spec.truncerr; rtol=1e-6)
    end

    @testset "svd truncation example 4" begin
      i = Index(QN(0,2)=>3,QN(1,2)=>4; tags="i")
      j = settags(i,"j")
      A = randomITensor(ElT, QN(1,2),i,j,dag(i'),dag(j'))

      maxdim = 4
      U,S,V,spec = svd(A,i,j; utags="x", vtags="y", maxdim=maxdim)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      u = commonind(S,U)
      v = commonind(S,V)

      @test hastags(u,"x")
      @test hastags(v,"y")

      @test hassameinds(U,(i,j,u))
      @test hassameinds(V,(i',j',v))

      for b in nzblocks(A)
        @test flux(A,b)==QN(1,2)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0,2)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(1,2)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0,2)
      end

      @test minimum(dims(S)) == maxdim
      @test minimum(dims(S)) == length(spec.eigs)
      @test minimum(dims(S)) < dim(i)*dim(j)

      Ap = U*S*V
      err = 1-(Ap*dag(Ap))[]/(A*dag(A))[]
      @test isapprox(err,spec.truncerr; rtol=1e-6)
    end

    @testset "svd truncation example 5" begin
      i = Index(QN(0,2)=>2,QN(1,2)=>3; tags="i")
      j = settags(i,"j")
      A = randomITensor(ElT, QN(1,2),i,j,dag(i'),dag(j'))

      maxdim = 4
      U,S,V,spec = svd(A,i,j'; utags="x", vtags="y", maxdim=maxdim)

      @test store(U) isa NDTensors.BlockSparse
      @test store(S) isa NDTensors.DiagBlockSparse
      @test store(V) isa NDTensors.BlockSparse

      u = commonind(S,U)
      v = commonind(S,V)

      @test hastags(u,"x")
      @test hastags(v,"y")

      @test hassameinds(U,(i,j',u))
      @test hassameinds(V,(i',j,v))

      for b in nzblocks(A)
        @test flux(A,b)==QN(1,2)
      end
      for b in nzblocks(U)
        @test flux(U,b)==QN(0,2)
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN(1,2)
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN(0,2)
      end

      @test minimum(dims(S)) == maxdim
      @test minimum(dims(S)) == length(spec.eigs)
      @test minimum(dims(S)) < dim(i)*dim(j)

      Ap = U*S*V
      err = 1-(Ap*dag(Ap))[]/(A*dag(A))[]
      @test isapprox(err,spec.truncerr; rtol=1e-6)
    end

    @testset "issue #231" begin
      l = Index(QN("Nf",-1,-1)=>2,
            QN("Nf", 0,-1)=>4,
            QN("Nf",+1,-1)=>2,
            tags="CMB,Link")
      s = Index(QN("Nf",0,-1)=>1,
                QN("Nf",1,-1)=>1,
                tags="Fermion,Site,n=4")
      r = Index(QN("Nf",1,-1)=>2,
                QN("Nf",0,-1)=>1,
                QN("Nf",1,-1)=>2,
                tags="Link,u")

      A = emptyITensor(ElT, l,s,dag(r))

      insertblock!(A,(2,1,2))
      insertblock!(A,(1,2,2))
      insertblock!(A,(2,2,3))

      for b in nzblocks(A)
        @test flux(A,b)==QN()
      end

      U,S,V = svd(A,l,s)

      for b in nzblocks(U)
        @test flux(U,b)==QN()
      end
      for b in nzblocks(S)
        @test flux(S,b)==QN()
      end
      for b in nzblocks(V)
        @test flux(V,b)==QN()
      end
      @test U * S * V ≈ A atol=1e-15
    end

    @testset "SVD no truncate bug" begin
      s = Index(QN("Sz",-4) => 1,
                QN("Sz",-2) => 4,
                QN("Sz", 0) => 6,
                QN("Sz", 2) => 4,
                QN("Sz", 4) => 1)
      A = emptyITensor(ElT, s, s')
      insertblock!(A, (5,2))
      insertblock!(A, (4,3))
      insertblock!(A, (3,4))
      insertblock!(A, (2,5))
      randn!(A)
      U,S,V = svd(A,s)
      @test U*S*V ≈ A
    end

    @testset "SVD no truncate" begin
      s = Index(QN("Sz",-4) => 1,
                QN("Sz",-2) => 4,
                QN("Sz", 0) => 6,
                QN("Sz", 2) => 4,
                QN("Sz", 4) => 1)
      A = emptyITensor(ElT, s, s')
      insertblock!(A, (5,1))
      insertblock!(A, (4,2))
      insertblock!(A, (3,3))
      insertblock!(A, (2,4))
      insertblock!(A, (1,5))
      U,S,V = svd(A, s)
      @test dims(S) == dims(A)
      @test U*S*V ≈ A
    end

    @testset "SVD truncate zeros" begin
      s = Index(QN("Sz",-4) => 1,
                QN("Sz",-2) => 4,
                QN("Sz", 0) => 6,
                QN("Sz", 2) => 4,
                QN("Sz", 4) => 1)
      A = emptyITensor(ElT, s, s')
      insertblock!(A, (5,1))
      insertblock!(A, (4,2))
      insertblock!(A, (3,3))
      insertblock!(A, (2,4))
      insertblock!(A, (1,5))
      U,S,V = svd(A, s; cutoff=0)
      @test dims(S) == (0,0)
      @test U*S*V ≈ A
    end

  end

  @testset "Replace Index" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    j = Index([QN(0)=>3,QN(1)=>4,QN(2)=>5],"j")

    T1 = randomITensor(QN(1),i,j)
    T2 = copy(T1)

    k = Index([QN(0)=>1,QN(1)=>2],"k")

    replaceind!(T1,i,k)
    @test hasind(T1,k)
    @test dir(inds(T1)[1]) == dir(i)

    # Check that replaceind! keeps
    # original Arrow direction
    replaceind!(T2,i,dag(k))
    @test hasind(T2,k)
    @test dir(inds(T2)[1]) == dir(i)
    @test dir(inds(T2)[1]) != dir(dag(k))
  end

@testset "BlockSparse dag copy behavior" begin
  i = Index(QN(0)=>2,QN(1)=>2,tags="i")
  j = Index(QN(0)=>2,QN(1)=>2,tags="j")

  v1 = randomITensor(QN(1),i,j)
  orig_elt = v1[1,3]
  cv1 = dag(v1;always_copy=false)
  cv1[1,3] = 123.45
  @test v1[1,3] ≈ cv1[1,3]

  v2 = randomITensor(QN(1),i,j)
  orig_elt = v2[1,3]
  cv2 = dag(v2;always_copy=true)
  cv2[1,3] = 123.45
  @test v2[1,3] ≈ orig_elt

  v3 = randomITensor(ComplexF64,QN(1),i,j)
  orig_elt = v3[1,3]
  cv3 = dag(v3;always_copy=false)
  cv3[1,3] = 123.45
  @test v3[1,3] ≈ orig_elt

  v4 = randomITensor(ComplexF64,QN(1),i,j)
  orig_elt = v4[1,3]
  cv4 = dag(v4;always_copy=true)
  cv4[1,3] = 123.45
  @test v4[1,3] ≈ orig_elt

end

@testset "exponentiate" begin

  @testset "Simple arrows" begin
    i1 = Index([QN(0)=>1,QN(1)=>2],"i1")
    i2 = Index([QN(0)=>1,QN(1)=>2],"i2")
    A = randomITensor(QN(),i1,i2,dag(i1)', dag(i2)')
    Aexp = exp(A)
    Amat = Array(A, i1,i2, i1', i2')
    Amatexp = reshape(exp(reshape(Amat,9,9)),3,3,3,3)
    @test Array(Aexp,i1,i2,i1',i2') ≈ Amatexp rtol=1e-14
    @test flux(Aexp) == QN()
    @test length(setdiff(inds(Aexp),inds(A)))==0

    @test exp(A, (i1,i2), (i1',i2')) ≈ Aexp rtol=5e-14

    # test the case where indices are permuted
    A = randomITensor(QN(),i1,dag(i1)', dag(i2)',i2)
    Aexp = exp(A, (i1,i2), (i1',i2'))
    Amat = Array(A, i1,i2,i1', i2')
    Amatexp = reshape(exp(reshape(Amat,9,9)),3,3,3,3)
    @test Array(Aexp,i1,i2,i1',i2') ≈ Amatexp rtol=1e-14

    # test exponentiation in the Hermitian case
    i1 = Index([QN(0)=>2,QN(1)=>2,QN(2)=>3],"i1")
    A = randomITensor(QN(),i1,dag(i1)')
    Ad = dag(swapinds(A,IndexSet(i1),IndexSet(dag(i1)')))
    Ah = A+Ad + 1e-10*randomITensor(QN(),i1,dag(i1)')
    Amat = Array(Ah ,i1', i1)
    Aexp = exp(Ah; ishermitian=true)
    Amatexp= exp(LinearAlgebra.Hermitian(Amat))
    @test Array(Aexp,i1,i1') ≈ Amatexp rtol=5e-14
  end

  @testset "Mixed arrows" begin
    i1 = Index([QN(0)=>1,QN(1)=>2],"i1")
    i2 = Index([QN(0)=>1,QN(1)=>2],"i2")
    A = randomITensor(i1, i2, dag(i1)', dag(i2)')
    expA = exp(A, (i1, i1'), (i2', i2))
    @test exp(dense(A), (i1, i1'), (i2', i2)) ≈ dense(expA)
  end

  @testset "Test contraction direction error" begin
    i = Index([QN(0)=>1, QN(1)=>1], "i")
    A = randomITensor(i', dag(i))
    A² = A' * A
    @test dense(A²) ≈ dense(A') * dense(A)
    @test_throws ErrorException A' * dag(A)
  end
end

end

nothing
