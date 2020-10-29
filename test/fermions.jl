using ITensors,
      Test

@testset "Fermions" begin

  @testset "parity_sign function" begin

    # Full permutations
    p1 = [1,2,3]
    @test ITensors.parity_sign(p1) == +1
    p2 = [2,1,3]
    @test ITensors.parity_sign(p2) == -1
    p3 = [2,3,1]
    @test ITensors.parity_sign(p3) == +1
    p4 = [3,2,1]
    @test ITensors.parity_sign(p4) == -1

    ## Partial permutations
    p5 = [2,7]
    @test ITensors.parity_sign(p5) == +1
    p6 = [5,3]
    @test ITensors.parity_sign(p6) == -1
    p7 = [1,9,3,10]
    @test ITensors.parity_sign(p7) == -1
    p8 = [1,12,9,3,11]
    @test ITensors.parity_sign(p8) == +1
  end

  @testset "Fermionic QNs" begin
    q = QN("Nf",1,-1)
    @test isfermionic(q[1])
    @test fparity(q) == 1

    q = q+q+q
    @test val(q,"Nf") == 3

    p = QN("P",1,-2)
    @test fparity(p) == 1
    @test fparity(p+p) == 0
    @test fparity(p+p+p) == 1
  end

  @testset "Fermionic IndexVals" begin
    sn = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")
    @test fparity(sn(1)) == 0
    @test fparity(sn(2)) == 1

    sp = Index([QN("Nfp",0,-2)=>1,QN("Nfp",1,-2)=>1],"sp")
    @test fparity(sp(1)) == 0
    @test fparity(sp(2)) == 1
  end

  @testset "Get and Set Elements" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

    N = ITensor(s',dag(s))

    N[s'(2),s(2)] = 1.0
    @test N[s'(2),s(2)] ≈ +1.0
    @test N[s(2),s'(2)] ≈ -1.0

    N[s(2),s'(2)] = 1.0
    @test N[s'(2),s(2)] ≈ -1.0
    @test N[s(2),s'(2)] ≈ 1.0

    C = ITensor(QN("Nf",-1,-1),s',dag(s))

    C[s'(1),s(2)] = 1.0
    @test C[s'(1),s(2)] ≈ 1.0
    @test C[s(2),s'(1)] ≈ 1.0


    I = ITensor(s',dag(s))
    I[s'(1),s(1)] = 1.0
    I[s'(2),s(2)] = 1.0
    @test I[s'(1),s(1)] ≈ 1.0
    @test I[s'(2),s(2)] ≈ 1.0

    @test I[s(1),s'(1)] ≈ 1.0
    @test I[s(2),s'(2)] ≈ -1.0
  end

  @testset "Making operators different ways" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

    N1 = ITensor(s',dag(s))
    N1[s'(2),s(2)] = +1.0

    N2 = ITensor(dag(s),s')
    N2[s'(2),s(2)] = +1.0
    @test norm(N1-N2) ≈ 0.0

    N3 = ITensor(s',dag(s))
    N3[s(2),s'(2)] = -1.0
    @test norm(N1-N3) ≈ 0.0

    N4 = ITensor(dag(s),s')
    N4[s(2),s'(2)] = -1.0
    @test norm(N1-N4) ≈ 0.0
  end

end

nothing
