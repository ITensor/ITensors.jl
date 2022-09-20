using ITensors, Test

@testset "Physics Sites" begin
  N = 10

  @testset "Generic sites" for eltype in (Float32, Float64, ComplexF32, ComplexF64)
    d1, d2 = 3, 4
    i1, i2 = Index(d1), Index(d2)

    o = op("I", i1; eltype)
    @test o == itensor(Matrix(I, d1, d1), i1', dag(i1))
    @test Base.eltype(o) <: eltype

    o = op("Id", i1; eltype)
    @test o == itensor(Matrix(I, d1, d1), i1', dag(i1))
    @test Base.eltype(o) <: eltype

    o = op("F", i1; eltype)
    @test o == itensor(Matrix(I, d1, d1), i1', dag(i1))
    @test Base.eltype(o) <: eltype

    o = op("I", i1, i2; eltype)
    @test o == itensor(Matrix(I, d1 * d2, d1 * d2), i2', i1', dag(i2), dag(i1))
    @test Base.eltype(o) <: eltype

    o = op("Id", i1, i2; eltype)
    @test o == itensor(Matrix(I, d1 * d2, d1 * d2), i2', i1', dag(i2), dag(i1))
    @test Base.eltype(o) <: eltype

    U1 = op("RandomUnitary", i1)
    @test hassameinds(U1, (i1', i1))
    @test apply(transpose(dag(U1)), U1) ≈ itensor(Matrix(I, d1, d1), i1', dag(i1))

    U12 = op("RandomUnitary", i1, i2)
    @test hassameinds(U12, (i1', i2', i1, i2))
    @test apply(transpose(dag(U12)), U12) ≈
          itensor(Matrix(I, d1 * d2, d1 * d2), i2', i1', dag(i2), dag(i1))
  end

  @testset "Qubit sites" begin
    s = siteind("Qubit")
    @test hastags(s, "Qubit,Site")
    @test dim(s) == 2

    s = siteinds("Qubit", N)
    @test val(s[1], "0") == 1
    @test val(s[1], "1") == 2
    @test_throws ArgumentError val(s[1], "Fake")

    s = siteind("Qubit"; conserve_parity=true)
    @test hastags(s, "Qubit,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN("Parity", 0, 2)
    @test qn(s, 2) == QN("Parity", 1, 2)

    s = siteind("Qubit"; conserve_number=true, conserve_parity=true)
    @test hastags(s, "Qubit,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN(("Parity", 0, 2), ("Number", 0))
    @test qn(s, 2) == QN(("Parity", 1, 2), ("Number", 1))

    s = siteinds("Qubit", N)

    Z = op("Z", s, 5)
    @test hasinds(Z, s[5]', s[5])

    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op("Id", s, 3), s[3]', s[3]) ≈ [1.0 0.0; 0.0 1.0]
    @test Array(op("√NOT", s, 3), s[3]', s[3]) ≈
          [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2]
    @test Array(op("H", s, 3), s[3]', s[3]) ≈ [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]
    @test Array(op("Phase", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("P", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("S", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("π/8", s, 3), s[3]', s[3]) ≈ [1 0; 0 (1 + im)/sqrt(2)]
    @test Array(op("T", s, 3), s[3]', s[3]) ≈ [1 0; 0 (1 + im)/sqrt(2)]
    θ = randn()
    @test Array(op("Rx", s, 3; θ=θ), s[3]', s[3]) ≈
          [cos(θ / 2) -im*sin(θ / 2); -im*sin(θ / 2) cos(θ / 2)]

    # Test obtaining S=1/2 operators using Qubit tag
    @test Array(op("X", s, 3), s[3]', s[3]) ≈ [0.0 1.0; 1.0 0.0]
  end

  @testset "Spin Half sites" begin
    s = siteind("S=1/2")
    @test hastags(s, "S=1/2,Site")
    @test dim(s) == 2

    s = siteind("S=1/2"; conserve_qns=true)
    @test hastags(s, "S=1/2,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN("Sz", +1)
    @test qn(s, 2) == QN("Sz", -1)

    s = siteind("S=1/2"; conserve_szparity=true)
    @test hastags(s, "S=1/2,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN("SzParity", 1, 2)
    @test qn(s, 2) == QN("SzParity", 0, 2)

    s = siteind("S=1/2"; conserve_sz=true, conserve_szparity=true)
    @test hastags(s, "S=1/2,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN(("SzParity", 1, 2), ("Sz", +1))
    @test qn(s, 2) == QN(("SzParity", 0, 2), ("Sz", -1))

    s = siteinds("S=1/2", N)
    @test val(s[1], "Up") == 1
    @test val(s[1], "↑") == 1
    @test val(s[1], "Dn") == 2
    @test val(s[1], "↓") == 2
    @test_throws ArgumentError val(s[1], "Fake")

    Sz5 = op("Sz", s, 5)
    @test hasinds(Sz5, s[5]', s[5])

    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op("Id", s, 3), s[3]', s[3]) ≈ [1.0 0.0; 0.0 1.0]
    @test Array(op("F", s, 3), s[3]', s[3]) ≈ [1.0 0.0; 0.0 1.0]
    @test Array(op("S+", s, 3), s[3]', s[3]) ≈ [0.0 1.0; 0.0 0.0]
    @test Array(op("S⁺", s, 3), s[3]', s[3]) ≈ [0.0 1.0; 0.0 0.0]
    @test Array(op("S-", s, 4), s[4]', s[4]) ≈ [0.0 0.0; 1.0 0.0]
    @test Array(op("S⁻", s, 4), s[4]', s[4]) ≈ [0.0 0.0; 1.0 0.0]
    @test Array(op("Sx", s, 2), s[2]', s[2]) ≈ [0.0 0.5; 0.5 0.0]
    @test Array(op("Sˣ", s, 2), s[2]', s[2]) ≈ [0.0 0.5; 0.5 0.0]
    @test Array(op("iSy", s, 2), s[2]', s[2]) ≈ [0.0 0.5; -0.5 0.0]
    @test Array(op("iSʸ", s, 2), s[2]', s[2]) ≈ [0.0 0.5; -0.5 0.0]
    @test Array(op("Sy", s, 2), s[2]', s[2]) ≈ [0.0 -0.5im; 0.5im 0.0]
    @test Array(op("Sʸ", s, 2), s[2]', s[2]) ≈ [0.0 -0.5im; 0.5im 0.0]
    @test Array(op("Sz", s, 2), s[2]', s[2]) ≈ [0.5 0.0; 0.0 -0.5]
    @test Array(op("Sᶻ", s, 2), s[2]', s[2]) ≈ [0.5 0.0; 0.0 -0.5]
    @test Array(op("ProjUp", s, 2), s[2]', s[2]) ≈ [1.0 0.0; 0.0 0.0]
    @test Array(op("projUp", s, 2), s[2]', s[2]) ≈ [1.0 0.0; 0.0 0.0]
    @test Array(op("ProjDn", s, 2), s[2]', s[2]) ≈ [0.0 0.0; 0.0 1.0]
    @test Array(op("projDn", s, 2), s[2]', s[2]) ≈ [0.0 0.0; 0.0 1.0]

    # Test obtaining Qubit operators using S=1/2 tag:
    @test Array(op("√NOT", s, 3), s[3]', s[3]) ≈
          [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2]
  end

  @testset "Spin One sites" begin
    s = siteinds("S=1", N)

    @test val(s[1], "Up") == 1
    @test val(s[1], "↑") == 1
    @test val(s[1], "0") == 2
    @test val(s[1], "Dn") == 3
    @test val(s[1], "↓") == 3
    @test_throws ArgumentError val(s[1], "Fake")

    Sz5 = op("Sz", s, 5)
    @test hasinds(Sz5, s[5]', s[5])

    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op("Id", s, 3), s[3]', s[3]) ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test Array(op("S+", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
    @test Array(op("S⁺", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
    @test Array(op("S-", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
    @test Array(op("S⁻", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
    @test Array(op("Sx", s, 3), s[3]', s[3]) ≈ [0 1/√2 0; 1/√2 0 1/√2; 0 1/√2 0]
    @test Array(op("Sˣ", s, 3), s[3]', s[3]) ≈ [0 1/√2 0; 1/√2 0 1/√2; 0 1/√2 0]
    @test Array(op("iSy", s, 3), s[3]', s[3]) ≈ [0 1/√2 0; -1/√2 0 1/√2; 0 -1/√2 0]
    @test Array(op("iSʸ", s, 3), s[3]', s[3]) ≈ [0 1/√2 0; -1/√2 0 1/√2; 0 -1/√2 0]
    @test Array(op("Sy", s, 3), s[3]', s[3]) ≈ (1 / (√2im)) * [0 +1 0; -1 0 +1; 0 -1 0]
    @test Array(op("Sʸ", s, 3), s[3]', s[3]) ≈ (1 / (√2im)) * [0 +1 0; -1 0 +1; 0 -1 0]
    #@test Array(op("Sʸ", s, 3), s[3]', s[3]) ≈ [0 +1/√2im 0; +1/√2im 0 -1/√2im; 0 +1/√2im 0]
    @test Array(op("Sz", s, 2), s[2]', s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 -1.0]
    @test Array(op("Sᶻ", s, 2), s[2]', s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 -1.0]
    @test Array(op("Sz2", s, 2), s[2]', s[2]) ≈ [1.0 0 0; 0 0 0; 0 0 +1.0]
    @test Array(op("Sx2", s, 2), s[2]', s[2]) ≈ [0.5 0 0.5; 0 1.0 0; 0.5 0 0.5]
    @test Array(op("Sy2", s, 2), s[2]', s[2]) ≈ [0.5 0 -0.5; 0 1.0 0; -0.5 0 0.5]
  end

  @testset "Fermion sites" begin
    s = siteind("Fermion")

    @test val(s, "0") == 1
    @test val(s, "1") == 2
    @test_throws ArgumentError val(s, "Fake")

    N = op(s, "N")
    @test hasinds(N, s', s)

    @test_throws ArgumentError op(s, "Fake")
    N = Array(op(s, "N"), s', s)
    @test N ≈ [0.0 0; 0 1]
    N = Array(op(s, "n"), s', s)
    @test N ≈ [0.0 0; 0 1]
    C = Array(op(s, "C"), s', s)
    @test C ≈ [0.0 1; 0 0]
    C = Array(op(s, "c"), s', s)
    @test C ≈ [0.0 1; 0 0]
    Cdag = Array(op(s, "Cdag"), s', s)
    @test Cdag ≈ [0.0 0; 1 0]
    Cdag = Array(op(s, "c†"), s', s)
    @test Cdag ≈ [0.0 0; 1 0]
    F = Array(op(s, "F"), s', s)
    @test F ≈ [1.0 0; 0 -1]

    @test has_fermion_string("C", s)
    @test has_fermion_string("c", s)
    @test has_fermion_string("Cdag", s)
    @test has_fermion_string("c†", s)
    @test has_fermion_string("C*F", s)
    @test has_fermion_string("c*F", s)
    @test has_fermion_string("F*Cdag*F", s)
    @test has_fermion_string("F*c†*F", s)
    @test !has_fermion_string("N", s)
    @test !has_fermion_string("n", s)
    @test !has_fermion_string("N*F", s)
    @test !has_fermion_string("n*F", s)

    s = siteind("Fermion"; conserve_nf=true)
    @test qn(s, 1) == QN("Nf", 0, -1)
    @test qn(s, 2) == QN("Nf", 1, -1)
    s = siteind("Fermion"; conserve_nfparity=true)
    @test qn(s, 1) == QN("NfParity", 0, -2)
    @test qn(s, 2) == QN("NfParity", 1, -2)
    s = siteind("Fermion"; conserve_qns=false)
    @test dim(s) == 2

    s = siteind("Fermion"; conserve_nf=true, conserve_sz=true)
    @test qn(s, 1) == QN(("Nf", 0, -1), ("Sz", 0))
    @test qn(s, 2) == QN(("Nf", 1, -1), ("Sz", 1))
    s = siteind("Fermion"; conserve_nfparity=true, conserve_sz=true)
    @test qn(s, 1) == QN(("NfParity", 0, -2), ("Sz", 0))
    @test qn(s, 2) == QN(("NfParity", 1, -2), ("Sz", 1))
    s = siteind("Fermion"; conserve_nf=true, conserve_sz="Up")
    @test qn(s, 1) == QN(("Nf", 0, -1), ("Sz", 0))
    @test qn(s, 2) == QN(("Nf", 1, -1), ("Sz", 1))
    s = siteind("Fermion"; conserve_nfparity=true, conserve_sz="Up")
    @test qn(s, 1) == QN(("NfParity", 0, -2), ("Sz", 0))
    @test qn(s, 2) == QN(("NfParity", 1, -2), ("Sz", 1))
    s = siteind("Fermion"; conserve_nf=true, conserve_sz="Dn")
    @test qn(s, 1) == QN(("Nf", 0, -1), ("Sz", 0))
    @test qn(s, 2) == QN(("Nf", 1, -1), ("Sz", -1))
    s = siteind("Fermion"; conserve_nfparity=true, conserve_sz="Dn")
    @test qn(s, 1) == QN(("NfParity", 0, -2), ("Sz", 0))
    @test qn(s, 2) == QN(("NfParity", 1, -2), ("Sz", -1))
  end

  @testset "Electron sites" begin
    s = siteind("Electron")

    @test val(s, "Emp") == 1
    @test val(s, "0") == 1
    @test val(s, "Up") == 2
    @test val(s, "↑") == 2
    @test val(s, "Dn") == 3
    @test val(s, "↓") == 3
    @test val(s, "UpDn") == 4
    @test val(s, "↑↓") == 4
    @test_throws ArgumentError val(s, "Fake")

    Nup = op(s, "Nup")
    @test hasinds(Nup, s', s)

    @test_throws ArgumentError op(s, "Fake")
    Nup = Array(op(s, "Nup"), s', s)
    @test Nup ≈ [0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
    Nup = Array(op(s, "n↑"), s', s)
    @test Nup ≈ [0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1]
    Ndn = Array(op(s, "Ndn"), s', s)
    @test Ndn ≈ [0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    Ndn = Array(op(s, "n↓"), s', s)
    @test Ndn ≈ [0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
    Ntot = Array(op(s, "Ntot"), s', s)
    @test Ntot ≈ [0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    Ntot = Array(op(s, "ntot"), s', s)
    @test Ntot ≈ [0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    Cup = Array(op(s, "Cup"), s', s)
    @test Cup ≈ [0.0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Cup = Array(op(s, "c↑"), s', s)
    @test Cup ≈ [0.0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Cdagup = Array(op(s, "Cdagup"), s', s)
    @test Cdagup ≈ [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Cdagup = Array(op(s, "c†↑"), s', s)
    @test Cdagup ≈ [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Cdn = Array(op(s, "Cdn"), s', s)
    @test Cdn ≈ [0.0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
    Cdn = Array(op(s, "c↓"), s', s)
    @test Cdn ≈ [0.0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
    Cdagdn = Array(op(s, "Cdagdn"), s', s)
    @test Cdagdn ≈ [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
    Cdagdn = Array(op(s, "c†↓"), s', s)
    @test Cdagdn ≈ [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
    F = Array(op(s, "F"), s', s)
    @test F ≈ [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
    Fup = Array(op(s, "Fup"), s', s)
    @test Fup ≈ [1.0 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    Fup = Array(op(s, "F↑"), s', s)
    @test Fup ≈ [1.0 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    Fdn3 = Array(op(s, "Fdn"), s', s)
    @test Fdn3 ≈ [1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    Fdn3 = Array(op(s, "F↓"), s', s)
    @test Fdn3 ≈ [1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    Sz3 = Array(op(s, "Sz"), s', s)
    @test Sz3 ≈ [0.0 0 0 0; 0 0.5 0 0; 0 0 -0.5 0; 0 0 0 0]
    Sz3 = Array(op(s, "Sᶻ"), s', s)
    @test Sz3 ≈ [0.0 0 0 0; 0 0.5 0 0; 0 0 -0.5 0; 0 0 0 0]
    Sx3 = Array(op(s, "Sx"), s', s)
    @test Sx3 ≈ [0.0 0 0 0; 0 0 0.5 0; 0 0.5 0 0; 0 0 0 0]
    Sx3 = Array(op(s, "Sˣ"), s', s)
    @test Sx3 ≈ [0.0 0 0 0; 0 0 0.5 0; 0 0.5 0 0; 0 0 0 0]
    Sp3 = Array(op(s, "S+"), s', s)
    @test Sp3 ≈ [0.0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sp3 = Array(op(s, "S⁺"), s', s)
    @test Sp3 ≈ [0.0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "S-"), s', s)
    @test Sm3 ≈ [0.0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "S⁻"), s', s)
    @test Sm3 ≈ [0.0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]

    @test has_fermion_string("Cup", s)
    @test has_fermion_string("c↑", s)
    @test has_fermion_string("Cup*F", s)
    @test has_fermion_string("c↑*F", s)
    @test has_fermion_string("Cdagup", s)
    @test has_fermion_string("c†↑", s)
    @test has_fermion_string("F*Cdagup", s)
    @test has_fermion_string("F*c†↑", s)
    @test has_fermion_string("Cdn", s)
    @test has_fermion_string("c↓", s)
    @test has_fermion_string("Cdn*F", s)
    @test has_fermion_string("c↓*F", s)
    @test has_fermion_string("Cdagdn", s)
    @test has_fermion_string("c†↓", s)
    @test !has_fermion_string("N", s)
    @test !has_fermion_string("n", s)
    @test !has_fermion_string("F*N", s)
    @test !has_fermion_string("F*n", s)

    s = siteind("Electron"; conserve_nf=true)
    @test qn(s, 1) == QN("Nf", 0, -1)
    @test qn(s, 2) == QN("Nf", 1, -1)
    @test qn(s, 3) == QN("Nf", 2, -1)
    s = siteind("Electron"; conserve_sz=true)
    @test qn(s, 1) == QN(("Sz", 0), ("NfParity", 0, -2))
    @test qn(s, 2) == QN(("Sz", +1), ("NfParity", 1, -2))
    @test qn(s, 3) == QN(("Sz", -1), ("NfParity", 1, -2))
    @test qn(s, 4) == QN(("Sz", 0), ("NfParity", 0, -2))
    s = siteind("Electron"; conserve_nfparity=true)
    @test qn(s, 1) == QN("NfParity", 0, -2)
    @test qn(s, 2) == QN("NfParity", 1, -2)
    @test qn(s, 3) == QN("NfParity", 0, -2)
    s = siteind("Electron"; conserve_qns=false)
    @test dim(s) == 4
  end

  @testset "tJ sites" begin
    s = siteind("tJ")

    @test val(s, "Emp") == 1
    @test val(s, "0") == 1
    @test val(s, "Up") == 2
    @test val(s, "↑") == 2
    @test val(s, "Dn") == 3
    @test val(s, "↓") == 3
    @test_throws ArgumentError val(s, "Fake")

    @test_throws ArgumentError op(s, "Fake")
    Nup = op(s, "Nup")
    @test Nup[2, 2] ≈ 1.0
    Nup = op(s, "n↑")
    @test Nup[2, 2] ≈ 1.0
    Ndn = op(s, "Ndn")
    @test Ndn[3, 3] ≈ 1.0
    Ndn = op(s, "n↓")
    @test Ndn[3, 3] ≈ 1.0
    Ntot = op(s, "Ntot")
    @test Ntot[2, 2] ≈ 1.0
    @test Ntot[3, 3] ≈ 1.0
    Ntot = op(s, "ntot")
    @test Ntot[2, 2] ≈ 1.0
    @test Ntot[3, 3] ≈ 1.0
    Id = Array(op(s, "Id"), s', s)
    @test Id ≈ [1.0 0 0; 0 1 0; 0 0 1]
    Cup = Array(op(s, "Cup"), s', s)
    @test Cup ≈ [0.0 1 0; 0 0 0; 0 0 0]
    Cup = Array(op(s, "c↑"), s', s)
    @test Cup ≈ [0.0 1 0; 0 0 0; 0 0 0]
    Cdup = Array(op(s, "Cdagup"), s', s)
    @test Cdup ≈ [0 0 0; 1.0 0 0; 0 0 0]
    Cdup = Array(op(s, "c†↑"), s', s)
    @test Cdup ≈ [0 0 0; 1.0 0 0; 0 0 0]
    Cdn = Array(op(s, "Cdn"), s', s)
    @test Cdn ≈ [0.0 0.0 1; 0 0 0; 0 0 0]
    Cdn = Array(op(s, "c↓"), s', s)
    @test Cdn ≈ [0.0 0.0 1; 0 0 0; 0 0 0]
    Cddn = Array(op(s, "Cdagdn"), s', s)
    @test Cddn ≈ [0 0 0; 0.0 0 0; 1 0 0]
    Cddn = Array(op(s, "c†↓"), s', s)
    @test Cddn ≈ [0 0 0; 0.0 0 0; 1 0 0]
    FP = Array(op(s, "F"), s', s)
    @test FP ≈ [1.0 0.0 0; 0 -1.0 0; 0 0 -1.0]
    Fup = Array(op(s, "Fup"), s', s)
    @test Fup ≈ [1.0 0.0 0; 0 -1.0 0; 0 0 1.0]
    Fup = Array(op(s, "F↑"), s', s)
    @test Fup ≈ [1.0 0.0 0; 0 -1.0 0; 0 0 1.0]
    Fdn = Array(op(s, "Fdn"), s', s)
    @test Fdn ≈ [1.0 0.0 0; 0 1.0 0; 0 0 -1.0]
    Fdn = Array(op(s, "F↓"), s', s)
    @test Fdn ≈ [1.0 0.0 0; 0 1.0 0; 0 0 -1.0]
    Sz = Array(op(s, "Sz"), s', s)
    @test Sz ≈ [0.0 0.0 0; 0 0.5 0; 0 0 -0.5]
    Sz = Array(op(s, "Sᶻ"), s', s)
    @test Sz ≈ [0.0 0.0 0; 0 0.5 0; 0 0 -0.5]
    Sx = Array(op(s, "Sx"), s', s)
    @test Sx ≈ [0.0 0.0 0; 0 0 0.5; 0 0.5 0]
    Sx = Array(op(s, "Sˣ"), s', s)
    @test Sx ≈ [0.0 0.0 0; 0 0 0.5; 0 0.5 0]
    Sp = Array(op(s, "Splus"), s', s)
    @test Sp ≈ [0.0 0.0 0; 0 0 1.0; 0 0 0]
    Sp = Array(op(s, "S⁺"), s', s)
    @test Sp ≈ [0.0 0.0 0; 0 0 1.0; 0 0 0]
    Sm = Array(op(s, "Sminus"), s', s)
    @test Sm ≈ [0.0 0.0 0; 0 0 0; 0 1.0 0]
    Sm = Array(op(s, "S⁻"), s', s)
    @test Sm ≈ [0.0 0.0 0; 0 0 0; 0 1.0 0]

    @test has_fermion_string("Cup", s)
    @test has_fermion_string("c↑", s)
    @test has_fermion_string("Cdagup", s)
    @test has_fermion_string("c†↑", s)
    @test has_fermion_string("Cdn", s)
    @test has_fermion_string("c↓", s)
    @test has_fermion_string("Cdagdn", s)
    @test has_fermion_string("c†↓", s)
    @test !has_fermion_string("N", s)
    @test !has_fermion_string("n", s)
  end

  @testset "$st" for st in ["Qudit", "Boson"]
    d = 3
    s = siteinds(st, 4; dim=d)
    @test dim(s[1]) == d
    @test dim(s[2]) == d
    @test dim(s[3]) == d
    @test dim(s[4]) == d
    v = state(s, 2, "0")
    @test v == itensor([1, 0, 0], s[2])
    v = state(s, 3, "1")
    @test v == itensor([0, 1, 0], s[3])
    v = state(s, 4, "2")
    @test v == itensor([0, 0, 1], s[4])
    @test_throws BoundsError state(s, 4, "3")
    v = val(s, 2, "0")
    @test v == 1
    v = val(s, 3, "1")
    @test v == 2
    v = val(s, 4, "2")
    @test v == 3
    @test op(s, "Id", 2) == itensor([1 0 0; 0 1 0; 0 0 1], s[2]', dag(s[2]))
    @test op(s, "I", 2) == itensor([1 0 0; 0 1 0; 0 0 1], s[2]', dag(s[2]))
    @test op(s, "F", 2) == itensor([1 0 0; 0 1 0; 0 0 1], s[2]', dag(s[2]))
    @test op("Id", s, 1, 2) ==
          itensor(Matrix(I, d^2, d^2), s[2]', s[1]', dag(s[2]), dag(s[1]))
    @test op("I", s, 1, 2) ==
          itensor(Matrix(I, d^2, d^2), s[2]', s[1]', dag(s[2]), dag(s[1]))
    @test op(s, "N", 2) == itensor([0 0 0; 0 1 0; 0 0 2], s[2]', dag(s[2]))
    @test op(s, "n", 2) == itensor([0 0 0; 0 1 0; 0 0 2], s[2]', dag(s[2]))
    @test op(s, "Adag", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "adag", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "a†", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "A", 2) ≈ itensor([0 1 0; 0 0 √2; 0 0 0], s[2]', dag(s[2]))
    @test op(s, "a", 2) ≈ itensor([0 1 0; 0 0 √2; 0 0 0], s[2]', dag(s[2]))

    # With QNs
    s = siteinds(st, 4; dim=d, conserve_qns=true)
    @test all(hasqns, s)
    @test op(s, "Id", 2) == itensor([1 0 0; 0 1 0; 0 0 1], s[2]', dag(s[2]))
    @test op(s, "I", 2) == itensor([1 0 0; 0 1 0; 0 0 1], s[2]', dag(s[2]))
    @test op("Id", s, 1, 2) ==
          itensor(Matrix(I, d^2, d^2), s[2]', s[1]', dag(s[2]), dag(s[1]))
    @test op("I", s, 1, 2) ==
          itensor(Matrix(I, d^2, d^2), s[2]', s[1]', dag(s[2]), dag(s[1]))
    @test op(s, "N", 2) == itensor([0 0 0; 0 1 0; 0 0 2], s[2]', dag(s[2]))
    @test op(s, "n", 2) == itensor([0 0 0; 0 1 0; 0 0 2], s[2]', dag(s[2]))
    @test op(s, "Adag", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "adag", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "a†", 2) ≈ itensor([0 0 0; 1 0 0; 0 √2 0], s[2]', dag(s[2]))
    @test op(s, "A", 2) ≈ itensor([0 1 0; 0 0 √2; 0 0 0], s[2]', dag(s[2]))
    @test op(s, "a", 2) ≈ itensor([0 1 0; 0 0 √2; 0 0 0], s[2]', dag(s[2]))
  end
end

nothing
