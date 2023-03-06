using ITensors, LinearAlgebra, Test

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

    U1 = op("randU", i1)
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

    s = siteind("Qubit"; conserve_number=true)
    @test hastags(s, "Qubit,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN("Number", 0)
    @test qn(s, 2) == QN("Number", 1)

    s = siteind("Qubit"; conserve_number=true, conserve_parity=true)
    @test hastags(s, "Qubit,Site")
    @test dim(s) == 2
    @test nblocks(s) == 2
    @test qn(s, 1) == QN(("Parity", 0, 2), ("Number", 0))
    @test qn(s, 2) == QN(("Parity", 1, 2), ("Number", 1))

    s = siteinds("Qubit", N)

    Z = op("Z", s, 5)
    @test hasinds(Z, s[5]', s[5])

    @test_throws ArgumentError(
      "Overload of \"state\" or \"state!\" functions not found for state name \"Fake\" and Index tags $(tags(s[3]))",
    ) state("Fake", s[3])
    @test Vector(state("Up", s[3])) ≈ [1, 0]
    @test Vector(state("↑", s[3])) ≈ [1, 0]
    @test Vector(state("Dn", s[3])) ≈ [0, 1]
    @test Vector(state("↓", s[3])) ≈ [0, 1]
    @test Vector(state("+", s[3])) ≈ (1 / √2) * [1, 1]
    @test Vector(state("X+", s[3])) ≈ (1 / √2) * [1, 1]
    @test Vector(state("Xp", s[3])) ≈ (1 / √2) * [1, 1]
    @test Vector(state("-", s[3])) ≈ (1 / √2) * [1, -1]
    @test Vector(state("X-", s[3])) ≈ (1 / √2) * [1, -1]
    @test Vector(state("Xm", s[3])) ≈ (1 / √2) * [1, -1]
    @test Vector(state("i", s[3])) ≈ (1 / √2) * [1, im]
    @test Vector(state("Yp", s[3])) ≈ (1 / √2) * [1, im]
    @test Vector(state("Y+", s[3])) ≈ (1 / √2) * [1, im]
    @test Vector(state("-i", s[3])) ≈ (1 / √2) * [1, -im]
    @test Vector(state("Y-", s[3])) ≈ (1 / √2) * [1, -im]
    @test Vector(state("Ym", s[3])) ≈ (1 / √2) * [1, -im]
    @test Vector(state("Z+", s[3])) ≈ [1, 0]
    @test Vector(state("Zp", s[3])) ≈ [1, 0]
    @test Vector(state("Z-", s[3])) ≈ [0, 1]
    @test Vector(state("Zm", s[3])) ≈ [0, 1]
    @test Vector(state("Tetra1", s[3])) ≈ [1, 0]
    @test Vector(state("Tetra2", s[3])) ≈ (1 / √3) * [1, √2]
    @test Vector(state("Tetra3", s[3])) ≈ (1 / √3) * [1, √2 * exp(im * 2π / 3)]
    @test Vector(state("Tetra4", s[3])) ≈ (1 / √3) * [1, √2 * exp(im * 4π / 3)]

    @test_throws ArgumentError op(s, "Fake", 2)
    @test Array(op("Id", s, 3), s[3]', s[3]) ≈ [1.0 0.0; 0.0 1.0]
    @test Array(op("√NOT", s, 3), s[3]', s[3]) ≈
      [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2]
    @test Array(op("√X", s, 3), s[3]', s[3]) ≈
      [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2]
    @test Array(op("σx", s, 3), s[3]', s[3]) ≈ [0 1; 1 0]
    @test Array(op("σ1", s, 3), s[3]', s[3]) ≈ [0 1; 1 0]
    @test Array(op("σy", s, 3), s[3]', s[3]) ≈ [0 -im; im 0]
    @test Array(op("σ2", s, 3), s[3]', s[3]) ≈ [0 -im; im 0]
    @test Array(op("iY", s, 3), s[3]', s[3]) ≈ [0 1; -1 0]
    @test Array(op("iσy", s, 3), s[3]', s[3]) ≈ [0 1; -1 0]
    @test Array(op("iσ2", s, 3), s[3]', s[3]) ≈ [0 1; -1 0]
    @test Array(op("σz", s, 3), s[3]', s[3]) ≈ [1 0; 0 -1]
    @test Array(op("σ3", s, 3), s[3]', s[3]) ≈ [1 0; 0 -1]
    @test Array(op("H", s, 3), s[3]', s[3]) ≈ [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]
    @test Array(op("Phase", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("P", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("S", s, 3), s[3]', s[3]) ≈ [1 0; 0 im]
    @test Array(op("π/8", s, 3), s[3]', s[3]) ≈ [1 0; 0 (1 + im)/sqrt(2)]
    @test Array(op("T", s, 3), s[3]', s[3]) ≈ [1 0; 0 (1 + im)/sqrt(2)]
    θ = randn()
    @test Array(op("Rx", s, 3; θ=θ), s[3]', s[3]) ≈
      [cos(θ / 2) -im*sin(θ / 2); -im*sin(θ / 2) cos(θ / 2)]
    @test Array(op("Ry", s, 3; θ=θ), s[3]', s[3]) ≈
      [cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
    @test Array(op("Rz", s, 3; θ=θ), s[3]', s[3]) ≈ [exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]
    # fallback
    @test Array(op("Rz", s, 3; ϕ=θ), s[3]', s[3]) ≈ [exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]
    λ = randn()
    φ = randn()
    @test Array(op("Rn", s, 3; θ=θ, λ=λ, ϕ=φ), s[3]', s[3]) ≈ [
      cos(θ / 2) -exp(im * λ)*sin(θ / 2)
      exp(im * φ)*sin(θ / 2) exp(im * (φ + λ))*cos(θ / 2)
    ]
    @test Array(op("Rn̂", s, 3; θ=θ, λ=λ, ϕ=φ), s[3]', s[3]) ≈ [
      cos(θ / 2) -exp(im * λ)*sin(θ / 2)
      exp(im * φ)*sin(θ / 2) exp(im * (φ + λ))*cos(θ / 2)
    ]
    @test Array(op("Splus", s, 3), s[3]', s[3]) ≈ [0 1; 0 0]
    @test Array(op("Sminus", s, 3), s[3]', s[3]) ≈ [0 0; 1 0]
    @test Array(op("S²", s, 3), s[3]', s[3]) ≈ [0.75 0; 0 0.75]
    @test Array(op("Proj0", s, 3), s[3]', s[3]) ≈ [1 0; 0 0]
    @test Array(op("Proj1", s, 3), s[3]', s[3]) ≈ [0 0; 0 1]
    @test reshape(Array(op("√SWAP", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1]
    @test reshape(Array(op("√Swap", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1]
    @test reshape(Array(op("√iSWAP", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 1/√2 im/√2 0; 0 im/√2 1/√2 0; 0 0 0 1]
    @test reshape(Array(op("√iSwap", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 1/√2 im/√2 0; 0 im/√2 1/√2 0; 0 0 0 1]
    @test reshape(Array(op("SWAP", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    @test reshape(Array(op("Swap", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    @test reshape(Array(op("iSWAP", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]
    @test reshape(Array(op("iSwap", s, 3, 5), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]
    @test reshape(Array(op("Cphase", s, 3, 5; ϕ=θ), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(im * θ)]
    @test reshape(Array(op("RXX", s, 3, 5; ϕ=θ), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈ [
      cos(θ) 0 0 -im*sin(θ)
      0 cos(θ) -im*sin(θ) 0
      0 -im*sin(θ) cos(θ) 0
      -im*sin(θ) 0 0 cos(θ)
    ]
    @test reshape(Array(op("RYY", s, 3, 5; ϕ=θ), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈ [
      cos(θ) 0 0 im*sin(θ)
      0 cos(θ) -im*sin(θ) 0
      0 -im*sin(θ) cos(θ) 0
      im*sin(θ) 0 0 cos(θ)
    ]
    @test reshape(Array(op("RXY", s, 3, 5; ϕ=θ), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [1 0 0 0; 0 cos(θ) -im*sin(θ) 0; 0 -im*sin(θ) cos(θ) 0; 0 0 0 1]
    @test reshape(Array(op("RZZ", s, 3, 5; ϕ=θ), s[3]', s[5]', s[3], s[5]), (4, 4)) ≈
      [exp(-im * θ) 0 0 0; 0 exp(im * θ) 0 0; 0 0 exp(im * θ) 0; 0 0 0 exp(-im * θ)]
    @test reshape(Array(op("CRX", s, 3, 5; θ=θ), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 cos(θ / 2) -im*sin(θ / 2)
      0 0 -im*sin(θ / 2) cos(θ / 2)
    ]
    @test reshape(Array(op("CRY", s, 3, 5; θ=θ), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 cos(θ / 2) -sin(θ / 2)
      0 0 sin(θ / 2) cos(θ / 2)
    ]
    @test reshape(Array(op("CRZ", s, 3, 5; θ=θ), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 exp(-im * θ / 2) 0
      0 0 0 exp(im * θ / 2)
    ]
    @test reshape(
      Array(op("CRn", s, 3, 5; θ=θ, λ=λ, ϕ=φ), s[5]', s[3]', s[5], s[3]), (4, 4)
    ) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 cos(θ / 2) -exp(im * λ)*sin(θ / 2)
      0 0 exp(im * φ)*sin(θ / 2) exp(im * (φ + λ))*cos(θ / 2)
    ]
    @test reshape(
      Array(op("CRn̂", s, 3, 5; θ=θ, λ=λ, ϕ=φ), s[5]', s[3]', s[5], s[3]), (4, 4)
    ) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 cos(θ / 2) -exp(im * λ)*sin(θ / 2)
      0 0 exp(im * φ)*sin(θ / 2) exp(im * (φ + λ))*cos(θ / 2)
    ]
    @test reshape(Array(op("CX", s, 3, 5), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 0 1
      0 0 1 0
    ]
    @test reshape(Array(op("CY", s, 3, 5), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 0 -im
      0 0 im 0
    ]
    @test reshape(Array(op("CZ", s, 3, 5), s[5]', s[3]', s[5], s[3]), (4, 4)) ≈ [
      1 0 0 0
      0 1 0 0
      0 0 1 0
      0 0 0 -1
    ]

    toff_mat = diagm(ones(8))
    toff_mat[7:8, 7:8] .= [0 1; 1 0]
    @test reshape(
      Array(op("TOFF", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ toff_mat
    @test reshape(
      Array(op("CCNOT", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ toff_mat
    @test reshape(
      Array(op("CCX", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ toff_mat
    fred_mat = diagm(ones(8))
    fred_mat[6:7, 6:7] .= [0 1; 1 0]
    @test reshape(
      Array(op("CS", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ fred_mat
    @test reshape(
      Array(op("CSWAP", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ fred_mat
    @test reshape(
      Array(op("CSwap", s, 3, 4, 5), s[5]', s[4]', s[3]', s[5], s[4], s[3]), (8, 8)
    ) ≈ fred_mat
    cccn_mat = diagm(ones(16))
    cccn_mat[15:16, 15:16] .= [0 1; 1 0]
    @test reshape(
      Array(
        op("CCCNOT", s, 2, 3, 4, 5), s[5]', s[4]', s[3]', s[2]', s[5], s[4], s[3], s[2]
      ),
      (16, 16),
    ) ≈ cccn_mat
    # Test obtaining S=1/2 operators using Qubit tag
    @test Matrix(op("X", s, 3), s[3]', s[3]) ≈ [0.0 1.0; 1.0 0.0]
  end

  @testset "Spin Half sites" begin
    for name in ("S=1/2", "SpinHalf", "S=½")
      s = siteind(name)
      @test hastags(s, name * ",Site")
      @test dim(s) == 2

      s = siteind(name; conserve_qns=true)
      @test hastags(s, name * ",Site")
      @test dim(s) == 2
      @test nblocks(s) == 2
      @test qn(s, 1) == QN("Sz", +1)
      @test qn(s, 2) == QN("Sz", -1)

      s = siteind(name; conserve_szparity=true)
      @test hastags(s, name * ",Site")
      @test dim(s) == 2
      @test nblocks(s) == 2
      @test qn(s, 1) == QN("SzParity", 1, 2)
      @test qn(s, 2) == QN("SzParity", 0, 2)

      s = siteind(name; conserve_sz=true, conserve_szparity=true)
      @test hastags(s, name * ",Site")
      @test dim(s) == 2
      @test nblocks(s) == 2
      @test qn(s, 1) == QN(("SzParity", 1, 2), ("Sz", +1))
      @test qn(s, 2) == QN(("SzParity", 0, 2), ("Sz", -1))

      s = siteinds(name, N)
      @test val(s[1], "Up") == 1
      @test val(s[1], "↑") == 1
      @test val(s[1], "Dn") == 2
      @test val(s[1], "↓") == 2
      @test_throws ArgumentError val(s[1], "Fake")

      Sz5 = op("Sz", s, 5)
      @test hasinds(Sz5, s[5]', s[5])

      @test Vector(state("Up", s[1])) ≈ [1, 0]
      @test Vector(state("↑", s[1])) ≈ [1, 0]
      @test Vector(state("Dn", s[1])) ≈ [0, 1]
      @test Vector(state("↓", s[1])) ≈ [0, 1]
      @test Vector(state("X+", s[1])) ≈ (1 / √2) * [1, 1]
      @test Vector(state("X-", s[1])) ≈ (1 / √2) * [1, -1]

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
  end

  @testset "Spin One sites" begin
    for name in ("S=1", "SpinOne")
      s = siteinds(name, N)

      @test val(s[1], "Up") == 1
      @test val(s[1], "↑") == 1
      @test val(s[1], "0") == 2
      @test val(s[1], "Dn") == 3
      @test val(s[1], "↓") == 3
      @test val(s[1], "Z+") == 1
      @test val(s[1], "Z-") == 3
      @test_throws ArgumentError val(s[1], "Fake")

      @test Vector(state("Up", s[1])) ≈ [1, 0, 0]
      @test Vector(state("↑", s[1])) ≈ [1, 0, 0]
      @test Vector(state("Z+", s[1])) ≈ [1, 0, 0]
      @test Vector(state("Z0", s[1])) ≈ [0, 1, 0]
      @test Vector(state("0", s[1])) ≈ [0, 1, 0]
      @test Vector(state("Dn", s[1])) ≈ [0, 0, 1]
      @test Vector(state("↓", s[1])) ≈ [0, 0, 1]
      @test Vector(state("Z-", s[1])) ≈ [0, 0, 1]
      @test Vector(state("X+", s[1])) ≈ [1 / 2, 1 / √2, 1 / 2]
      @test Vector(state("X0", s[1])) ≈ [-1 / √2, 0, 1 / √2]
      @test Vector(state("X-", s[1])) ≈ [1 / 2, -1 / √2, 1 / 2]
      @test Vector(state("Y+", s[1])) ≈ [-1 / 2, -im / √2, 1 / 2]
      @test Vector(state("Y0", s[1])) ≈ [1 / √2, 0, 1 / √2]
      @test Vector(state("Y-", s[1])) ≈ [-1 / 2, im / √2, 1 / 2]

      Sz5 = op("Sz", s, 5)
      @test hasinds(Sz5, s[5]', s[5])

      @test_throws ArgumentError op(s, "Fake", 2)
      @test Array(op("Id", s, 3), s[3]', s[3]) ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
      @test Array(op("S+", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
      @test Array(op("S⁺", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
      @test Array(op("Sp", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
      @test Array(op("Splus", s, 3), s[3]', s[3]) ≈ [0 √2 0; 0 0 √2; 0 0 0]
      @test Array(op("S-", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
      @test Array(op("S⁻", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
      @test Array(op("Sm", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
      @test Array(op("Sminus", s, 3), s[3]', s[3]) ≈ [0 0 0; √2 0 0; 0.0 √2 0]
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
  end

  @testset "Fermion sites" begin
    s = siteind("Fermion")

    @test val(s, "0") == 1
    @test val(s, "1") == 2
    @test_throws ArgumentError val(s, "Fake")

    @test Vector(state("Emp", s)) ≈ [1, 0]
    @test Vector(state("Occ", s)) ≈ [0, 1]
    @test Vector(state("0", s)) ≈ [1, 0]
    @test Vector(state("1", s)) ≈ [0, 1]

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
    Cdag = Array(op(s, "cdag"), s', s)
    @test Cdag ≈ [0.0 0; 1 0]
    Cdag = Array(op(s, "c†"), s', s)
    @test Cdag ≈ [0.0 0; 1 0]
    F = Array(op(s, "F"), s', s)
    @test F ≈ [1.0 0; 0 -1]

    @test has_fermion_string("C", s)
    @test has_fermion_string("c", s)
    @test has_fermion_string("Cdag", s)
    @test has_fermion_string("cdag", s)
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
    s = siteind("Fermion"; conserve_parity=true)
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

    @test Vector(state("Emp", s)) ≈ [1, 0, 0, 0]
    @test Vector(state("Up", s)) ≈ [0, 1, 0, 0]
    @test Vector(state("Dn", s)) ≈ [0, 0, 1, 0]
    @test Vector(state("UpDn", s)) ≈ [0, 0, 0, 1]
    @test Vector(state("0", s)) ≈ [1, 0, 0, 0]
    @test Vector(state("↑", s)) ≈ [0, 1, 0, 0]
    @test Vector(state("↓", s)) ≈ [0, 0, 1, 0]
    @test Vector(state("↑↓", s)) ≈ [0, 0, 0, 1]

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
    Nupdn = Array(op(s, "n↑↓"), s', s)
    @test Nupdn ≈ [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
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
    Aup = Array(op(s, "Aup"), s', s)
    @test Aup ≈ [0.0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Aup = Array(op(s, "a↑"), s', s)
    @test Aup ≈ [0.0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    Adagup = Array(op(s, "Adagup"), s', s)
    @test Adagup ≈ [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Adagup = Array(op(s, "a†↑"), s', s)
    @test Adagup ≈ [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
    Adn = Array(op(s, "Adn"), s', s)
    @test Adn ≈ [0.0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Adn = Array(op(s, "a↓"), s', s)
    @test Adn ≈ [0.0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Adagdn = Array(op(s, "Adagdn"), s', s)
    @test Adagdn ≈ [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
    Adagdn = Array(op(s, "a†↓"), s', s)
    @test Adagdn ≈ [0.0 0 0 0; 0 0 0 0; 1 0 0 0; 0 1 0 0]
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
    Sp3 = Array(op(s, "Sp"), s', s)
    @test Sp3 ≈ [0.0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sp3 = Array(op(s, "Splus"), s', s)
    @test Sp3 ≈ [0.0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sp3 = Array(op(s, "S⁺"), s', s)
    @test Sp3 ≈ [0.0 0 0 0; 0 0 1 0; 0 0 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "S-"), s', s)
    @test Sm3 ≈ [0.0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "S⁻"), s', s)
    @test Sm3 ≈ [0.0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "Sm"), s', s)
    @test Sm3 ≈ [0.0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]
    Sm3 = Array(op(s, "Sminus"), s', s)
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
    s = siteind("Electron"; conserve_parity=true)
    @test qn(s, 1) == QN("NfParity", 0, -2)
    @test qn(s, 2) == QN("NfParity", 1, -2)
    @test qn(s, 3) == QN("NfParity", 0, -2)
    s = siteind("Electron"; conserve_qns=false)
    @test dim(s) == 4
  end

  @testset "tJ sites" begin
    s = siteind("tJ"; conserve_parity=true)
    @test hastags(s, "tJ,Site")
    @test dim(s) == 3
    @test nblocks(s) == 2
    @test qn(s, 1) == QN(("NfParity", 0, -2))
    @test qn(s, 2) == QN(("NfParity", 1, -2))

    s = siteind("tJ"; conserve_nf=true)
    @test hastags(s, "tJ,Site")
    @test dim(s) == 3
    @test nblocks(s) == 2
    @test qn(s, 1) == QN(("Nf", 0, -1))
    @test qn(s, 2) == QN(("Nf", 1, -1))

    s = siteind("tJ"; conserve_sz=true)
    @test hastags(s, "tJ,Site")
    @test dim(s) == 3
    @test nblocks(s) == 3
    @test qn(s, 1) == QN(("Sz", 0), ("NfParity", 0, -2))
    @test qn(s, 2) == QN(("Sz", 1), ("NfParity", 1, -2))
    @test qn(s, 3) == QN(("Sz", -1), ("NfParity", 1, -2))

    s = siteind("tJ"; conserve_sz=true, conserve_nf=true)
    @test hastags(s, "tJ,Site")
    @test dim(s) == 3
    @test nblocks(s) == 3
    @test qn(s, 1) == QN(("Nf", 0, -1), ("Sz", 0))
    @test qn(s, 2) == QN(("Nf", 1, -1), ("Sz", 1))
    @test qn(s, 3) == QN(("Nf", 1, -1), ("Sz", -1))

    s = siteind("tJ")
    @test hastags(s, "tJ,Site")
    @test dim(s) == 3

    @test val(s, "Emp") == 1
    @test val(s, "0") == 1
    @test val(s, "Up") == 2
    @test val(s, "↑") == 2
    @test val(s, "Dn") == 3
    @test val(s, "↓") == 3
    @test_throws ArgumentError val(s, "Fake")

    @test Vector(state("Emp", s)) ≈ [1, 0, 0]
    @test Vector(state("0", s)) ≈ [1, 0, 0]
    @test Vector(state("Up", s)) ≈ [0, 1, 0]
    @test Vector(state("↑", s)) ≈ [0, 1, 0]
    @test Vector(state("Dn", s)) ≈ [0, 0, 1]
    @test Vector(state("↓", s)) ≈ [0, 0, 1]

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
    Aup = Array(op(s, "Aup"), s', s)
    @test Aup ≈ [0.0 1 0; 0 0 0; 0 0 0]
    Aup = Array(op(s, "a↑"), s', s)
    @test Aup ≈ [0.0 1 0; 0 0 0; 0 0 0]
    Adup = Array(op(s, "Adagup"), s', s)
    @test Adup ≈ [0 0 0; 1.0 0 0; 0 0 0]
    Adup = Array(op(s, "a†↑"), s', s)
    @test Adup ≈ [0 0 0; 1.0 0 0; 0 0 0]
    Adn = Array(op(s, "Adn"), s', s)
    @test Adn ≈ [0.0 0.0 1; 0 0 0; 0 0 0]
    Adn = Array(op(s, "a↓"), s', s)
    @test Adn ≈ [0.0 0.0 1; 0 0 0; 0 0 0]
    Addn = Array(op(s, "Adagdn"), s', s)
    @test Addn ≈ [0 0 0; 0.0 0 0; 1 0 0]
    Addn = Array(op(s, "a†↓"), s', s)
    @test Addn ≈ [0 0 0; 0.0 0 0; 1 0 0]
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
    Sp = Array(op(s, "Sp"), s', s)
    @test Sp ≈ [0.0 0.0 0; 0 0 1.0; 0 0 0]
    Sp = Array(op(s, "S⁺"), s', s)
    @test Sp ≈ [0.0 0.0 0; 0 0 1.0; 0 0 0]
    Sm = Array(op(s, "Sminus"), s', s)
    @test Sm ≈ [0.0 0.0 0; 0 0 0; 0 1.0 0]
    Sm = Array(op(s, "Sm"), s', s)
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
    @test op(s, "a†b†", 2, 3) ≈ itensor(
      kron([0 0 0; 1 0 0; 0 √2 0], [0 0 0; 1 0 0; 0 √2 0]),
      s[3]',
      s[2]',
      dag(s[3]),
      dag(s[2]),
    )
    @test op(s, "a†b", 2, 3) ≈ itensor(
      kron([0 0 0; 1 0 0; 0 √2 0], [0 1 0; 0 0 √2; 0 0 0]),
      s[3]',
      s[2]',
      dag(s[3]),
      dag(s[2]),
    )
    @test op(s, "ab†", 2, 3) ≈ itensor(
      kron([0 1 0; 0 0 √2; 0 0 0], [0 0 0; 1 0 0; 0 √2 0]),
      s[3]',
      s[2]',
      dag(s[3]),
      dag(s[2]),
    )
    @test op(s, "ab", 2, 3) ≈ itensor(
      kron([0 1 0; 0 0 √2; 0 0 0], [0 1 0; 0 0 √2; 0 0 0]),
      s[3]',
      s[2]',
      dag(s[3]),
      dag(s[2]),
    )
    @test_throws ErrorException op(ITensors.OpName("ab"), ITensors.SiteType(st))

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
