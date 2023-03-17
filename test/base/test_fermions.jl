using ITensors, Test

@testset "Fermions" begin
  ITensors.enable_auto_fermion()

  @testset "parity_sign function" begin

    # Full permutations
    p1 = [1, 2, 3]
    @test ITensors.parity_sign(p1) == +1
    p2 = [2, 1, 3]
    @test ITensors.parity_sign(p2) == -1
    p3 = [2, 3, 1]
    @test ITensors.parity_sign(p3) == +1
    p4 = [3, 2, 1]
    @test ITensors.parity_sign(p4) == -1

    ## Partial permutations
    p5 = [2, 7]
    @test ITensors.parity_sign(p5) == +1
    p6 = [5, 3]
    @test ITensors.parity_sign(p6) == -1
    p7 = [1, 9, 3, 10]
    @test ITensors.parity_sign(p7) == -1
    p8 = [1, 12, 9, 3, 11]
    @test ITensors.parity_sign(p8) == +1
  end

  @testset "Fermionic QNs" begin
    q = QN("Nf", 1, -1)
    @test isfermionic(q[1])
    @test fparity(q) == 1

    q = q + q + q
    @test val(q, "Nf") == 3

    p = QN("P", 1, -2)
    @test fparity(p) == 1
    @test isodd(p)
    @test fparity(p + p) == 0
    @test fparity(p + p + p) == 1
  end

  @testset "Fermionic IndexVals" begin
    sn = Index([QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1], "sn")
    @test fparity(sn => 1) == 0
    @test fparity(sn => 2) == 1
    @test !isodd(sn => 1)
    @test isodd(sn => 2)

    sp = Index([QN("Nfp", 0, -2) => 1, QN("Nfp", 1, -2) => 1], "sp")
    @test fparity(sp => 1) == 0
    @test fparity(sp => 2) == 1
  end

  @testset "Get and Set Elements" begin
    s = Index([QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1], "s")

    N = ITensor(s', dag(s))

    N[s' => 2, s => 2] = 1.0
    @test N[s' => 2, s => 2] ≈ +1.0
    @test N[s => 2, s' => 2] ≈ -1.0

    N[s => 2, s' => 2] = 1.0
    @test N[s' => 2, s => 2] ≈ -1.0
    @test N[s => 2, s' => 2] ≈ 1.0

    C = ITensor(s', dag(s))

    C[s' => 1, s => 2] = 1.0
    @test C[s' => 1, s => 2] ≈ 1.0
    @test C[s => 2, s' => 1] ≈ 1.0

    I = ITensor(s', dag(s))
    I[s' => 1, s => 1] = 1.0
    I[s' => 2, s => 2] = 1.0
    @test I[s' => 1, s => 1] ≈ 1.0
    @test I[s' => 2, s => 2] ≈ 1.0

    @test I[s => 1, s' => 1] ≈ 1.0
    @test I[s => 2, s' => 2] ≈ -1.0
  end

  @testset "Making operators different ways" begin
    s = Index([QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1], "s")

    N1 = ITensor(s', dag(s))
    N1[s' => 2, s => 2] = +1.0

    N2 = ITensor(dag(s), s')
    N2[s' => 2, s => 2] = +1.0
    @test norm(N1 - N2) ≈ 0.0

    N3 = ITensor(s', dag(s))
    N3[s => 2, s' => 2] = -1.0
    @test norm(N1 - N3) ≈ 0.0

    N4 = ITensor(dag(s), s')
    N4[s => 2, s' => 2] = -1.0
    @test norm(N1 - N4) ≈ 0.0
  end

  @testset "Permute and Add Fermionic ITensors" begin
    @testset "Permute Operators" begin
      s = Index([QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1], "s")

      N1 = ITensor(s', dag(s))
      N1[s' => 2, s => 2] = 1.0

      N2 = ITensor(dag(s), s')
      N2[s' => 2, s => 2] = 1.0

      pN1 = permute(N1, dag(s), s')
      @test pN1[s' => 2, s => 2] ≈ 1.0

      pN2 = permute(N2, s', dag(s))
      @test pN2[s' => 2, s => 2] ≈ 1.0

      #TODO add cases resulting in minus signs
    end

    @testset "Add Operators" begin
      s = Index([QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1], "sn")

      N1 = ITensor(s', dag(s))
      N1[s' => 2, s => 2] = 1.0

      N2 = ITensor(dag(s), s')
      N2[s' => 2, s => 2] = 1.0

      NN = N1 + N2
      @test NN[s' => 2, s => 2] ≈ 2.0

      NN = N1 + N1
      @test NN[s' => 2, s => 2] ≈ 2.0

      NN = N2 + N2
      @test NN[s' => 2, s => 2] ≈ 2.0
    end

    @testset "Wavefunction Tests" begin
      s = [Index([QN("N", 0, -2) => 2, QN("N", 1, -2) => 2], "s$n") for n in 1:4]

      psi0 = ITensor(s...)

      psi0[s[1] => 1, s[2] => 1, s[3] => 1, s[4] => 1] = 1111
      psi0[s[1] => 3, s[2] => 3, s[3] => 1, s[4] => 1] = 3311
      psi0[s[1] => 1, s[2] => 3, s[3] => 1, s[4] => 3] = 1313

      psi1 = permute(psi0, s[2], s[1], s[3], s[4])
      @test norm(psi1 - psi0) ≈ 0.0

      @test psi0[s[1] => 1, s[2] => 1, s[3] => 1, s[4] => 1] ≈ 1111
      @test psi1[s[1] => 1, s[2] => 1, s[3] => 1, s[4] => 1] ≈ 1111
      @test psi0[s[2] => 1, s[1] => 1, s[3] => 1, s[4] => 1] ≈ 1111
      @test psi1[s[2] => 1, s[1] => 1, s[3] => 1, s[4] => 1] ≈ 1111

      @test psi0[s[1] => 3, s[2] => 3, s[3] => 1, s[4] => 1] ≈ 3311
      @test psi1[s[1] => 3, s[2] => 3, s[3] => 1, s[4] => 1] ≈ 3311
      @test psi0[s[2] => 3, s[1] => 3, s[3] => 1, s[4] => 1] ≈ -3311
      @test psi1[s[2] => 3, s[1] => 3, s[3] => 1, s[4] => 1] ≈ -3311
      @test psi0[s[4] => 1, s[2] => 3, s[1] => 3, s[3] => 1] ≈ -3311
      @test psi1[s[4] => 1, s[2] => 3, s[1] => 3, s[3] => 1] ≈ -3311

      psi2 = permute(psi0, s[4], s[1], s[3], s[2])
      @test norm(psi2 - psi0) ≈ 0.0
      @test norm(psi2 - psi1) ≈ 0.0

      @test psi0[s[1] => 1, s[2] => 3, s[3] => 1, s[4] => 3] ≈ 1313
      @test psi1[s[1] => 1, s[2] => 3, s[3] => 1, s[4] => 3] ≈ 1313
      @test psi2[s[1] => 1, s[2] => 3, s[3] => 1, s[4] => 3] ≈ 1313
      @test psi0[s[4] => 3, s[1] => 1, s[3] => 1, s[2] => 3] ≈ -1313
      @test psi1[s[4] => 3, s[1] => 1, s[3] => 1, s[2] => 3] ≈ -1313
      @test psi2[s[4] => 3, s[1] => 1, s[3] => 1, s[2] => 3] ≈ -1313
    end
  end

  @testset "C Cdag operators" begin
    s = siteinds("Fermion", 3; conserve_qns=true)

    p110 = ITensor(s[1], s[2], s[3])
    p110[s[1] => 2, s[2] => 2, s[3] => 1] = 1.0

    p011 = ITensor(s[1], s[2], s[3])
    p011[s[1] => 1, s[2] => 2, s[3] => 2] = 1.0

    np011 = ITensor(s[1], s[2], s[3])
    np011[s[1] => 1, s[3] => 2, s[2] => 2] = 1.0

    dag_p011 = ITensor(dag(s[3]), dag(s[2]), dag(s[1]))
    dag_p011[s[3] => 2, s[2] => 2, s[1] => 1] = 1.0

    @test norm(dag(p011) - dag_p011) ≈ 0

    C1 = op(s, "C", 1)
    Cdag3 = op(s, "Cdag", 3)

    # Anti-commutator
    @test norm(Cdag3 * C1 + C1 * Cdag3) ≈ 0.0

    # Commutator
    @test norm(Cdag3 * C1 - C1 * Cdag3) ≈ 2.0

    let # <011|Cdag3*C1|110> = -1
      t1 = noprime(C1 * p110)
      t2 = noprime(Cdag3 * t1)
      @test scalar(dag_p011 * t2) ≈ -1.0
    end

    let # <011|C1*Cdag3|110> = +1
      t1 = noprime(Cdag3 * p110)
      t2 = noprime(C1 * t1)
      @test scalar(dag_p011 * t2) ≈ +1.0
    end

    let # <011|(Cdag3*C1)|110> = -1
      t = noprime((Cdag3 * C1) * p110)
      @test scalar(dag(p011) * t) ≈ -1.0
    end

    let # <011|(C1*Cdag3)|110> = +1
      t = noprime((C1 * Cdag3) * p110)
      @test scalar(dag(p011) * t) ≈ +1.0
    end

    #
    # Commuting B tensors
    #
    # These commute by carrying additional
    # g-indices (Grassman indices)
    # 

    g = Index(QN("Nf", 1, -1) => 1; tags="g")

    Bdag3 = Cdag3 * setelt(dag(g) => 1)
    B1 = setelt(g => 1) * C1

    # Commutator
    @test norm(Bdag3 * B1 - B1 * Bdag3) ≈ 0.0

    # Anti-commutator
    @test norm(Bdag3 * B1 + B1 * Bdag3) ≈ 2.0

    let # <011|Cdag3*C1|110> = <011|Bdag3*B1|110> = -1
      t1 = noprime(B1 * p110)
      t2 = noprime(Bdag3 * t1)
      @test scalar(dag(p011) * t2) ≈ -1.0
    end

    let # <011|(Cdag3*C1)|110> = <011|(Bdag3*B1)|110> = -1
      t = noprime((Bdag3 * B1) * p110)
      @test scalar(dag(p011) * t) ≈ -1.0
    end

    let # <011|Cdag3*C1|110> = <011|B1*Bdag3|110> = -1
      t1 = noprime(Bdag3 * p110)
      t2 = noprime(B1 * t1)
      @test scalar(dag(p011) * t2) ≈ -1.0
    end

    let # <011|(Cdag3*C1)|110> = <011|(B1*Bdag3)|110> = -1
      t = noprime((B1 * Bdag3) * p110)
      @test scalar(dag(p011) * t) ≈ -1.0
    end

    #
    # Leave out middle fermion, test for cases <001|...|100>
    #
    p100 = ITensor(s[1], s[2], s[3])
    p100[s[1] => 2, s[2] => 1, s[3] => 1] = 1.0

    p001 = ITensor(s[1], s[2], s[3])
    p001[s[1] => 1, s[2] => 1, s[3] => 2] = 1.0

    let # <001|Cdag3*C1|100> = <001|Bdag3*B1|100> = +1
      t1 = noprime(B1 * p100)
      t2 = noprime(Bdag3 * t1)
      @test scalar(dag(p001) * t2) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|(Bdag3*B1)|100> = +1
      t = noprime((Bdag3 * B1) * p100)
      @test scalar(dag(p001) * t) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|B1*Bdag3|100> = +1
      t1 = noprime(Bdag3 * p100)
      t2 = noprime(B1 * t1)
      @test scalar(dag(p001) * t2) ≈ +1.0
    end

    let # <001|Cdag3*C1|100> = <001|(B1*Bdag3)|100> = +1
      t = noprime((B1 * Bdag3) * p100)
      @test scalar(dag(p001) * t) ≈ +1.0
    end
  end

  @testset "Combiner conjugation" begin
    s = siteinds("Fermion", 4; conserve_qns=true)
    C = combiner(s[1], s[2])
    @test NDTensors.isconj(storage(C)) == false

    dC = dag(C)
    @test NDTensors.isconj(storage(dC)) == true
  end

  @testset "Combine Uncombine Permute Test" begin
    s = siteinds("Fermion", 4; conserve_qns=true)

    @testset "Two Site Test" begin
      p11 = ITensor(s[1], s[2])
      p11[s[1] => 2, s[2] => 2] = 1.0

      C = combiner(s[1], s[2])

      dp11 = dag(p11)

      Cp11_A = C * p11
      dCp11_A = dag(Cp11_A)
      dp11_A = C * dCp11_A
      @test dp11_A ≈ dp11

      Cp11_B = p11 * C
      dCp11_B = dag(Cp11_B)
      dp11_B = C * dCp11_B
      @test dp11_B ≈ dp11
    end

    @testset "Longer two-site tests" begin
      s1, s2, s3, s4 = s
      C12 = combiner(s1, s2)
      C21 = combiner(s2, s1)
      C13 = combiner(s1, s3)
      C31 = combiner(s3, s1)

      T = randomITensor(QN("Nf", 3, -1), s1, s2, s3, s4)
      T .= abs.(T)

      #
      # 1a, 2a tests
      #

      c12 = combinedind(C12)
      c12T = C12 * T
      u12T = dag(C12) * c12T
      @test norm(u12T - T) < 1E-10

      c21 = combinedind(C21)
      c21T = C21 * T
      u21T = dag(C21) * c21T
      @test norm(u21T - T) < 1E-10

      c13 = combinedind(C13)
      c13T = C13 * T
      u13T = dag(C13) * c13T
      @test norm(u13T - T) < 1E-10

      c31 = combinedind(C31)
      c31T = C31 * T
      u31T = dag(C31) * c31T
      @test norm(u31T - T) < 1E-10

      #
      # 1b, 2b tests
      #

      dc12T = dag(C12) * dag(T)
      @test norm(dc12T - dag(c12T)) < 1E-10
      du12T = C12 * dc12T
      @test norm(du12T - dag(T)) < 1E-10

      dc21T = dag(C21) * dag(T)
      @test norm(dc21T - dag(c21T)) < 1E-10
      du21T = C21 * dc21T
      @test norm(du21T - dag(T)) < 1E-10

      dc13T = dag(C13) * dag(T)
      @test norm(dc13T - dag(c13T)) < 1E-10
      du13T = C13 * dc13T
      @test norm(du13T - dag(T)) < 1E-10

      dc31T = dag(C31) * dag(T)
      @test norm(dc31T - dag(c31T)) < 1E-10
      du31T = C31 * dc31T
      @test norm(du31T - dag(T)) < 1E-10
    end

    @testset "Three Site Test" begin
      p111 = ITensor(s[1], s[2], s[3])
      p111[s[1] => 2, s[2] => 2, s[3] => 2] = 1.0

      dp111 = dag(p111)

      C = combiner(s[1], s[3])
      Cp111 = C * p111
      dCp111 = dag(Cp111)
      dp111_U = C * dCp111
      @test dp111_U ≈ dp111
    end
  end

  @testset "Mixed Arrow Combiner Tests" begin
    @testset "One wrong-way arrow" begin
      q1 = QN("Nf", 1, -1)

      s0 = Index([q1 => 1]; tags="s0")
      s1 = Index([q1 => 1]; tags="s1")
      s2 = Index([q1 => 1]; tags="s2")
      s3 = Index([q1 => 1]; tags="s3")
      s4 = Index([q1 => 1]; tags="s4")

      A = randomITensor(QN("Nf", 0, -1), s0, s1, dag(s2), dag(s3))
      B = randomITensor(QN("Nf", 0, -1), s3, s2, dag(s1), dag(s4))
      A .= one.(A)
      B .= one.(B)
      @test norm(A) ≈ 1.0
      @test norm(B) ≈ 1.0

      Ru = A * B

      C = combiner(s3, s2, dag(s1))
      Bc = C * B
      Ac = A * dag(C)
      Rc = Ac * Bc

      @test norm(Ru - Rc) < 1E-8
    end

    @testset "Two wrong-way arrows" begin
      q1 = QN("Nf", 1, -1)

      s0 = Index([q1 => 1]; tags="s0")
      s1 = Index([q1 => 1]; tags="s1")
      s2 = Index([q1 => 1]; tags="s2")
      s3 = Index([q1 => 1]; tags="s3")
      s4 = Index([q1 => 1]; tags="s4")

      A = randomITensor(QN("Nf", 2, -1), s0, s1, s2, dag(s3))
      B = randomITensor(QN("Nf", -2, -1), s3, dag(s2), dag(s1), dag(s4))
      A .= one.(A)
      B .= one.(B)
      @test norm(A) ≈ 1.0
      @test norm(B) ≈ 1.0

      Ru = A * B

      C = combiner(s3, dag(s2), dag(s1))
      Bc = C * B
      Ac = A * dag(C)
      Rc = Ac * Bc

      @test norm(Ru - Rc) < 1E-8
    end
  end

  @testset "Permutedims Regression Test" begin
    s1 = Index([QN("N", 0, -1) => 1, QN("N", 1, -1) => 1], "s1")
    s2 = Index([QN("N", 0, -1) => 1, QN("N", 1, -1) => 1], "s2")
    i = Index([QN("N", 0, -1) => 1, QN("N", 1, -1) => 1, QN("N", 2, -1) => 1], "i")

    A = ITensor(QN("N", 4, -1), s1, s2, i)
    A[s1 => 2, s2 => 2, i => 3] = 223

    B = ITensor(QN("N", 4, -1), s1, i, s2)
    B[s1 => 2, i => 3, s2 => 2] = 223
    @test A ≈ B

    C = ITensor(QN("N", 4, -1), s1, i, s2)
    C[s2 => 2, i => 3, s1 => 2] = -223
    @test A ≈ C
  end

  @testset "Fermionic SVD" begin
    N = 4
    s = siteinds("Fermion", N; conserve_qns=true)

    A = randomITensor(QN("Nf", 2, -1), s[1], s[2], s[3], s[4])
    for n1 in 1:4, n2 in 1:4
      (n1 == n2) && continue
      U, S, V = svd(A, (s[n1], s[n2]))
      @test norm(U * S * V - A) < 1E-10
    end
    for n1 in 1:4, n2 in 1:4, n3 in 1:4
      (n1 == n2) && continue
      (n1 == n3) && continue
      (n2 == n3) && continue
      U, S, V = svd(A, (s[n1], s[n2], s[n3]))
      @test norm(U * S * V - A) < 1E-10
    end

    B = randomITensor(QN("Nf", 3, -1), s[1], s[2], s[3], s[4])
    for n1 in 1:4, n2 in 1:4
      (n1 == n2) && continue
      U, S, V = svd(B, (s[n1], s[n2]))
      @test norm(U * S * V - B) < 1E-10
    end
    for n1 in 1:4, n2 in 1:4, n3 in 1:4
      (n1 == n2) && continue
      (n1 == n3) && continue
      (n2 == n3) && continue
      U, S, V = svd(B, (s[n1], s[n2], s[n3]))
      @test norm(U * S * V - B) < 1E-10
    end
  end # Fermionic SVD tests

  @testset "Fermion Contraction with Combined Indices" begin
    N = 10
    s = siteinds("Fermion", N; conserve_qns=true)

    begin
      A = randomITensor(QN("Nf", 3, -1), s[1], s[2], s[3], s[4])
      B = randomITensor(QN("Nf", 2, -1), s[1], s[3], s[4])

      CC = combiner(s[1], s[3])

      cA = CC * A
      cB = CC * B

      R1 = dag(cA) * cB
      R2 = dag(A) * B

      @test norm(R1 - R2) < 1E-10
    end

    begin
      A = randomITensor(QN("Nf", 3, -1), s[1], s[2], s[3], s[4])
      B = randomITensor(QN("Nf", 2, -1), s[1], s[3], s[4])

      CC = combiner(s[1], s[3])

      cA = CC * A
      cdB = dag(CC) * dag(B)

      R1 = cA * cdB
      R2 = A * dag(B)

      @test norm(R1 - R2) < 1E-10
    end

    begin
      A = randomITensor(QN("Nf", 3, -1), s[1], s[2], s[3], s[4])
      B = randomITensor(QN("Nf", 2, -1), s[1], s[3], s[4])

      CC = combiner(s[1], s[4], s[3])

      cA = CC * A
      cdB = dag(CC) * dag(B)

      R1 = cA * cdB
      R2 = A * dag(B)

      @test norm(R1 - R2) < 1E-10
    end

    begin
      CC = combiner(s[3], s[4])
      c = combinedind(CC)

      A = randomITensor(QN("Nf", 3, -1), c, s[1], s[2])
      B = randomITensor(QN("Nf", 2, -1), s[1], c, s[5])

      uA = dag(CC) * A
      uB = dag(CC) * B

      R1 = dag(uA) * uB
      R2 = dag(A) * B

      @test norm(R1 - R2) < 1E-10
    end

    @testset "Combiner Regression Test" begin
      T = randomITensor(QN("Nf", 3, -1), s[1], s[2], s[3], s[4])

      C12 = combiner(s[1], s[2])
      c12 = combinedind(C12)
      c12T = C12 * T

      u12T = dag(C12) * c12T

      @test norm(u12T - T) < 1E-10
    end
  end # Fermion Contraction with Combined Indices

  @testset "Regression Tests" begin
    @testset "SVD DiagBlockSparse Regression Test" begin
      l1 = Index(QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1; tags="Link,l=1")
      s2 = Index(QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1; tags="Site,n=2")
      s3 = Index(QN("Nf", 0, -1) => 1, QN("Nf", 1, -1) => 1; tags="Site,n=3")
      l3 = Index(QN("Nf", 2, -1) => 1; tags="Link,l=3")

      phi = randomITensor(QN("Nf", 4, -1), l1, s2, s3, l3)

      U, S, V = svd(phi, (l1, s2))

      @test norm((U * S) * V - phi) < 1E-10
      @test norm(U * (S * V) - phi) < 1E-10
    end

    @testset "Eigen Positive Semi Def Regression Test" begin
      #
      # Test was failing without using combiners in
      # eigen which were conjugates of each other
      #
      cutoff = 1E-12
      N = 2
      s = siteinds("Fermion", N; conserve_qns=true)

      T = ITensor(QN("Nf", 0, -1), dag(s[1]), s[1]')
      T[2, 2] = 1

      F = eigen(T; ishermitian=true, cutoff=cutoff)
      D, U, spec = F
      Ut = F.Vt

      @test norm(dag(U) * D * Ut - T) < 1E-10
    end

    @testset "Factorize Eigen Regression Test" begin
      N = 3
      s = siteinds("Fermion", N; conserve_qns=true)
      A = ITensor(QN("Nf", 2, -1), s[1], s[2], s[3])
      A[s[1] => 1, s[2] => 2, s[3] => 2] = 1.0

      U, R = factorize(A, (s[1], s[2]); which_decomp="eigen", cutoff=1E-18, ortho="left")

      @test norm(U * R - A) < 1E-12
    end

    @testset "Contraction Regression Test" begin
      s = siteinds("Fermion", 3; conserve_qns=true)
      l = Index(QN("Nf", 1, -1) => 1; tags="l")

      q2 = QN("Nf", 2, -1)
      q0 = QN("Nf", 0, -1)

      T1 = ITensor(q2, s[1], s[2], l)
      T1[s[1] => 1, s[2] => 2, l => 1] = 1.0

      T2 = ITensor(q0, dag(l), s[3])
      T2[dag(l) => 1, s[3] => 2] = 1.0

      @test norm(T1 * T2 - T2 * T1) < 1E-10
    end

    @testset "SVD Regression Test" begin
      Pf0 = QN("Pf", 0, -2)
      Pf1 = QN("Pf", 1, -2)

      l22 = Index([Pf0 => 1, Pf1 => 1], "Link,dir=2,n=2")
      l23 = Index([Pf0 => 1, Pf1 => 1], "Link,dir=3,n=2")
      s1 = Index([Pf0 => 1, Pf1 => 1, Pf1 => 1, Pf0 => 1], "Site,n=1")
      l11 = Index([Pf0 => 1, Pf1 => 1], "Link,dir=1,n=1")

      T = randomITensor(dag(l22), dag(l23), s1, l11)

      U, S, V = svd(T, dag(l22), dag(l23), s1)

      @test norm(T - U * S * V) < 1E-10
    end
  end # Regression Tests

  ITensors.disable_auto_fermion()
end

nothing
