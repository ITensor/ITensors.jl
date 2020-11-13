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

    N = emptyITensor(s',dag(s))

    N[s'(2),s(2)] = 1.0
    @test N[s'(2),s(2)] ≈ +1.0
    @test N[s(2),s'(2)] ≈ -1.0

    N[s(2),s'(2)] = 1.0
    @test N[s'(2),s(2)] ≈ -1.0
    @test N[s(2),s'(2)] ≈ 1.0

    C = emptyITensor(s',dag(s))

    C[s'(1),s(2)] = 1.0
    @test C[s'(1),s(2)] ≈ 1.0
    @test C[s(2),s'(1)] ≈ 1.0


    I = emptyITensor(s',dag(s))
    I[s'(1),s(1)] = 1.0
    I[s'(2),s(2)] = 1.0
    @test I[s'(1),s(1)] ≈ 1.0
    @test I[s'(2),s(2)] ≈ 1.0

    @test I[s(1),s'(1)] ≈ 1.0
    @test I[s(2),s'(2)] ≈ -1.0
  end

  @testset "Making operators different ways" begin
    s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

    N1 = emptyITensor(s',dag(s))
    N1[s'(2),s(2)] = +1.0

    N2 = emptyITensor(dag(s),s')
    N2[s'(2),s(2)] = +1.0
    @test norm(N1-N2) ≈ 0.0

    N3 = emptyITensor(s',dag(s))
    N3[s(2),s'(2)] = -1.0
    @test norm(N1-N3) ≈ 0.0

    N4 = emptyITensor(dag(s),s')
    N4[s(2),s'(2)] = -1.0
    @test norm(N1-N4) ≈ 0.0
  end

  @testset "Permute and Add Fermionic ITensors" begin

    @testset "Permute Operators" begin
      s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"s")

      N1 = emptyITensor(s',dag(s))
      N1[s'(2),s(2)] = 1.0

      N2 = emptyITensor(dag(s),s')
      N2[s'(2),s(2)] = 1.0

      pN1 = permute(N1,dag(s),s')
      @test pN1[s'(2),s(2)] ≈ 1.0

      pN2 = permute(N2,s',dag(s))
      @test pN2[s'(2),s(2)] ≈ 1.0

      #TODO add cases resulting in minus signs
    end

    @testset "Add Operators" begin
      s = Index([QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1],"sn")

      N1 = emptyITensor(s',dag(s))
      N1[s'(2),s(2)] = 1.0

      N2 = emptyITensor(dag(s),s')
      N2[s'(2),s(2)] = 1.0

      NN = N1+N2
      @test NN[s'(2),s(2)] ≈ 2.0

      NN = N1+N1
      @test NN[s'(2),s(2)] ≈ 2.0

      NN = N2+N2
      @test NN[s'(2),s(2)] ≈ 2.0
    end

    @testset "Wavefunction Tests" begin
      s = [Index([QN("N",0,-2)=>2,QN("N",1,-2)=>2],"s$n") for n=1:4]

      psi0 = ITensor(s...)

      psi0[s[1](1),s[2](1),s[3](1),s[4](1)] = 1111
      psi0[s[1](3),s[2](3),s[3](1),s[4](1)] = 3311
      psi0[s[1](1),s[2](3),s[3](1),s[4](3)] = 1313

      psi1 = permute(psi0,s[2],s[1],s[3],s[4])
      @test norm(psi1-psi0) ≈ 0.0

      @test psi0[s[1](1),s[2](1),s[3](1),s[4](1)] ≈ 1111
      @test psi1[s[1](1),s[2](1),s[3](1),s[4](1)] ≈ 1111
      @test psi0[s[2](1),s[1](1),s[3](1),s[4](1)] ≈ 1111
      @test psi1[s[2](1),s[1](1),s[3](1),s[4](1)] ≈ 1111

      @test psi0[s[1](3),s[2](3),s[3](1),s[4](1)] ≈ 3311
      @test psi1[s[1](3),s[2](3),s[3](1),s[4](1)] ≈ 3311
      @test psi0[s[2](3),s[1](3),s[3](1),s[4](1)] ≈ -3311
      @test psi1[s[2](3),s[1](3),s[3](1),s[4](1)] ≈ -3311
      @test psi0[s[4](1),s[2](3),s[1](3),s[3](1)] ≈ -3311
      @test psi1[s[4](1),s[2](3),s[1](3),s[3](1)] ≈ -3311

      psi2 = permute(psi0,s[4],s[1],s[3],s[2])
      @test norm(psi2-psi0) ≈ 0.0
      @test norm(psi2-psi1) ≈ 0.0

      @test psi0[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi1[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi2[s[1](1),s[2](3),s[3](1),s[4](3)] ≈ 1313
      @test psi0[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
      @test psi1[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
      @test psi2[s[4](3),s[1](1),s[3](1),s[2](3)] ≈ -1313
    end

  end

  @testset "C Cdag operators" begin
    s = siteinds("Fermion",3;conserve_qns=true)

    p110 = emptyITensor(s[1],s[2],s[3])
    p110[s[1]=>2,s[2]=>2,s[3]=>1] = 1.0

    p011 = emptyITensor(s[1],s[2],s[3])
    p011[s[1]=>1,s[2]=>2,s[3]=>2] = 1.0

    np011 = emptyITensor(s[1],s[2],s[3])
    np011[s[1]=>1,s[3]=>2,s[2]=>2] = 1.0

    dag_p011 = emptyITensor(dag(s[3]),dag(s[2]),dag(s[1]))
    dag_p011[s[3]=>2,s[2]=>2,s[1]=>1] = 1.0

    @test norm(dag(p011) - dag_p011) ≈ 0

    C1 = op(s,"C",1)
    Cdag3 = op(s,"Cdag",3)

    # Anti-commutator
    @test norm(Cdag3*C1 + C1*Cdag3) ≈ 0.0

    # Commutator
    @test norm(Cdag3*C1 - C1*Cdag3) ≈ 2.0

    let # <011|Cdag3*C1|110> = -1
      t1 = noprime(C1*p110)
      t2 = noprime(Cdag3*t1)
      @test scalar(dag_p011*t2) ≈ -1.0
    end

    let # <011|C1*Cdag3|110> = +1
      t1 = noprime(Cdag3*p110)
      t2 = noprime(C1*t1)
      @test scalar(dag_p011*t2) ≈ +1.0
    end

    let # <011|(Cdag3*C1)|110> = -1
      t = noprime((Cdag3*C1)*p110)
      @test scalar(dag(p011)*t) ≈ -1.0
    end

    let # <011|(C1*Cdag3)|110> = +1
      t = noprime((C1*Cdag3)*p110)
      @test scalar(dag(p011)*t) ≈ +1.0
    end

    #
    # Commuting B tensors
    #
    # These commute by carrying additional
    # g-indices (Grassman indices)
    # 
    
    g = Index(QN("Nf",1,-1)=>1,tags="g")

    Bdag3 = Cdag3*setelt(dag(g)(1))
    B1 = setelt(g(1))*C1 

    # Commutator
    @test norm(Bdag3*B1 - B1*Bdag3) ≈ 0.0

    # Anti-commutator
    @test norm(Bdag3*B1 + B1*Bdag3) ≈ 2.0

    let # <011|Cdag3*C1|110> = <011|Bdag3*B1|110> = -1
      t1 = noprime(B1*p110)
      t2 = noprime(Bdag3*t1)
      @test scalar(dag(p011)*t2) ≈ -1.0
    end

    let # <011|(Cdag3*C1)|110> = <011|(Bdag3*B1)|110> = -1
      t = noprime((Bdag3*B1)*p110)
      @test scalar(dag(p011)*t) ≈ -1.0
    end

    let # <011|Cdag3*C1|110> = <011|B1*Bdag3|110> = -1
      t1 = noprime(Bdag3*p110)
      t2 = noprime(B1*t1)
      @test scalar(dag(p011)*t2) ≈ -1.0
    end

    let # <011|(Cdag3*C1)|110> = <011|(B1*Bdag3)|110> = -1
      t = noprime((B1*Bdag3)*p110)
      @test scalar(dag(p011)*t) ≈ -1.0
    end

    #
    # Leave out middle fermion, test for cases <001|...|100>
    #
    p100 = emptyITensor(s[1],s[2],s[3])
    p100[s[1]=>2,s[2]=>1,s[3]=>1] = 1.0

    p001 = emptyITensor(s[1],s[2],s[3])
    p001[s[1]=>1,s[2]=>1,s[3]=>2] = 1.0

    let # <001|Cdag3*C1|100> = <001|Bdag3*B1|100> = +1
      t1 = noprime(B1*p100)
      t2 = noprime(Bdag3*t1)
      @test scalar(dag(p001)*t2) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|(Bdag3*B1)|100> = +1
      t = noprime((Bdag3*B1)*p100)
      @test scalar(dag(p001)*t) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|B1*Bdag3|100> = +1
      t1 = noprime(Bdag3*p100)
      t2 = noprime(B1*t1)
      @test scalar(dag(p001)*t2) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|(B1*Bdag3)|100> = +1
      t = noprime((B1*Bdag3)*p100)
      @test scalar(dag(p001)*t) ≈ +1.0
    end
  end

  @testset "Combine Uncombine Permute Test" begin
    s = siteinds("Fermion",3;conserve_qns=true)

    @testset "Two Site Test" begin
      p11 = emptyITensor(s[1],s[2])
      p11[s[1]=>2,s[2]=>2] = 1.0

      C = combiner(s[1],s[2])

      dp11 = dag(p11)

      Cp11_A = C*p11
      dCp11_A = dag(Cp11_A)
      dp11_A = C*dCp11_A
      @test dp11_A ≈ dp11

      Cp11_B = p11*C
      dCp11_B = dag(Cp11_B)
      dp11_B = C*dCp11_B
      @test dp11_B ≈ dp11
    end

    @testset "Three Site Test" begin
      p111 = emptyITensor(s[1],s[2],s[3])
      p111[s[1]=>2,s[2]=>2,s[3]=>2] = 1.0

      dp111 = dag(p111)

      C = combiner(s[1],s[3])
      Cp111 = C*p111
      dCp111 = dag(Cp111)
      dp111_U = C*dCp111
      @test dp111_U ≈ dp111
    end
  end

  @testset "Permutedims Regression Test" begin
    s1 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s1")
    s2 = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1],"s2")
    i = Index([QN("N",0,-1)=>1,QN("N",1,-1)=>1,QN("N",2,-1)=>1],"i")

    A = ITensor(QN("N",4,-1),s1,s2,i)
    A[s1[2],s2[2],i[3]] = 223

    B = ITensor(QN("N",4,-1),s1,i,s2)
    B[s1[2],i[3],s2[2]] = 223
    @test A ≈ B

    C = ITensor(QN("N",4,-1),s1,i,s2)
    C[s2[2],i[3],s1[2]] = -223
    @test A ≈ C
  end

  @testset "Product MPS consistency checks" begin
    s = siteinds("Fermion",3;conserve_qns=true)

    pA = productMPS(s,[2,1,2])
    TA = emptyITensor(s[1],s[2],s[3])
    TA[s[1]=>2,s[2]=>1,s[3]=>2] = 1.0
    A = pA[1]*pA[2]*pA[3]
    @test norm(A-TA) < 1E-8

    pB = productMPS(s,[1,2,2])
    TB = emptyITensor(s[1],s[2],s[3])
    TB[s[1]=>1,s[2]=>2,s[3]=>2] = 1.0
    B = pB[1]*pB[2]*pB[3]
    @test norm(B-TB) < 1E-8
  end

  @testset "Dag regression test" begin
    s = Index(QN("Nf",0,-1)=>1,QN("Nf",1,-1)=>1,tags="s")
    l1 = Index(QN("Nf",1,-1)=>1,tags="l1")
    l2 = Index(QN("Nf",2,-1)=>1,tags="l2")
    T = ITensor(l1,s,l2)
    T[1,2,1] = 1.0
    dT = dag(T)
    @test dT[1,2,1] ≈ 1.0
  end

  @testset "MPS inner regression test" begin
    sites = siteinds("Fermion",3;conserve_qns=true)
    psi = productMPS(sites,[2,2,1])
    @test inner(psi,psi) ≈ 1.0
  end

  @testset "Fermionic AutoMPO Tests" begin

    @testset "Spinless Fermion Hamiltonian" begin
      N = 2
      sites = siteinds("Fermion",N;conserve_qns=true)
      t1 = 1.0
      ampo = AutoMPO()
      for b=1:N-1
        ampo += -t1, "Cdag", b,  "C", b+1
        ampo += -t1, "Cdag", b+1,"C", b
      end
      H = MPO(ampo,sites)

      HH = H[1]
      for n=2:N
        HH *= H[n]
      end
      HHc = dag(swapprime(HH,0,1))
      @test norm(HHc-HH) < 1E-8
    end

    @testset "Fermion Hamiltonian Matrix Elements" begin
      N = 10
      t1 = 0.654
      V1 = 1.23

      sites = siteinds("Fermion",N;conserve_qns=true)

      ampo = AutoMPO()
      for b=1:N-1
        ampo += -t1,"Cdag",b,  "C",b+1
        ampo += -t1,"Cdag",b+1,"C",b
        ampo += V1,"N",b,"N",b+1
      end
      H = MPO(ampo,sites)

      for j=1:N-2
        stateA = [1 for n=1:N]
        stateA[j] = 2
        stateA[N] = 2 # to make MPS bosonic

        stateB = [1 for n=1:N]
        stateB[j+1] = 2
        stateB[N] = 2 # to make MPS bosonic

        psiA = productMPS(sites,stateA)
        psiB = productMPS(sites,stateB)

        @test inner(psiA,H,psiB) ≈ -t1
        @test inner(psiB,H,psiA) ≈ -t1
      end


      for j=1:N-1
        state = [1 for n=1:N]
        state[j] = 2
        state[j+1] = 2
        psi = productMPS(sites,state)
        @test inner(psi,H,psi) ≈ V1
      end
    end

  end

end

nothing
