using ITensors, Test

function is_unitary(U::ITensor; kwargs...)
  s = noprime(filterinds(U; plev=1))
  return isapprox(transpose(dag(U))(U), op("I", s...))
end

@testset "SiteType" begin
  N = 10

  @testset "Star in operator strings" begin
    @test_throws ErrorException op("S=1/2")

    sites = siteinds("S=1/2", N)
    #@test_throws ArgumentError op(sites, "Sp", 1)
    @test sites[1] isa Index
    Sz = op(sites, "Sz", 2)
    SzSz = op(sites, "Sz * Sz", 2)
    @test SzSz ≈ product(Sz, Sz)
    Sy = op(sites, "Sy", 2)
    SySy = op(sites, "Sy * Sy", 2)
    @test SySy ≈ product(Sy, Sy)

    Sz1 = op("Sz", sites, 1)
    @test op("Sz", [sites[1]]) ≈ Sz1
    @test op([sites[1]], "Sz") ≈ Sz1
    @test op([1 0; 0 -1] / 2, [sites[1]]) ≈ Sz1
    @test op([sites[1]], [1 0; 0 -1] / 2) ≈ Sz1

    @test op([sites[1]], "Ry"; θ=π / 2) ≈
      itensor([1 -1; 1 1] / √2, sites[1]', dag(sites[1]))

    sites = siteinds("S=1", N)
    #@test_throws ArgumentError op(sites, "Sp", 1)
    Sz = op(sites, "Sz", 2)
    SzSz = op(sites, "Sz * Sz", 2)
    @test SzSz ≈ product(Sz, Sz)
    Sy = op(sites, "Sy", 2)
    SySy = op(sites, "Sy * Sy", 2)
    @test SySy ≈ product(Sy, Sy)
    SzSySz = op(sites, "Sz * Sy * Sz", 2)
    @test SzSySz ≈ product(Sz, product(Sy, Sz))
  end

  @testset "+/- in operator strings" begin
    q = siteind("Qudit"; dim=5)
    Amat = array(op("a", q))
    Adagmat = array(op("a†", q))

    x = Amat - Adagmat
    @test x ≈ array(op("a - a†", q))
    x = Amat * Adagmat - Adagmat
    @test x ≈ array(op("a * a† - a†", q))
    @test x ≈ array(op("a * a† - a†", q))
    x = Adagmat * Adagmat * Amat * Amat
    @test x ≈ array(op("a† * a† * a * a", q))

    q = siteind("S=1/2")
    Sp = array(op("S+", q))
    Sm = array(op("S-", q))
    Sx = array(op("Sx", q))
    Sy = array(op("Sy", q))
    Sz = array(op("Sz", q))
    x = Sp + Sm
    @test x ≈ array(op("S+ + S-", q))
    x = Sp - Sm
    @test x ≈ array(op("S+ - S-", q))
    x = Sp - Sm - Sp
    @test x ≈ array(op("S+ - S- - S+", q))
    x = Sp * Sm + Sm * Sp
    @test x ≈ array(op("S+ * S- + S- * S+", q))
    # Deprecated syntax
    @test x ≈ array(op("S+ * S- + S-*S+", q))
    x = Sp * Sm - Sm * Sp
    @test x ≈ array(op("S+ * S- - S- * S+", q))
    @test x ≈ array(op("S+ * S- - S- * S+", q))
    x = Sp * Sm + Sm * Sp + Sz * Sx * Sy
    @test x ≈ array(op("S+ * S- + S- * S+ + Sz * Sx * Sy", q))
    x = Sp * Sm - Sm * Sp + Sz * Sx * Sy
    @test x ≈ array(op("S+ * S- - S- * S+ + Sz * Sx * Sy", q))
    x = Sp * Sm - Sm * Sp - Sz * Sx * Sy
    @test x ≈ array(op("S+ * S- - S- * S+ - Sz * Sx * Sy", q))

    #q = siteind("Qubit")
    #R = array(op("Rx", q; θ = 0.1))
    #H = array(op("H", q))
    #Y = array(op("Y", q))
    #x = H * R + Y + R
    #@test x ≈ array(op("H * Rx + Y + Rx", q; θ = 0.1))

  end

  @testset "Custom SiteType using op" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op(::OpName"Sz", ::SiteType"_Custom_", s::Index)
      Op = emptyITensor(s', dag(s))
      Op[s' => 1, s => 1] = +3 / 2
      Op[s' => 2, s => 2] = +1 / 2
      Op[s' => 3, s => 3] = -1 / 2
      Op[s' => 4, s => 4] = -3 / 2
      return Op
    end

    function ITensors.op(::OpName"α", ::SiteType"_Custom_", s1::Index, s2::Index)
      Op = emptyITensor(s1', s2', dag(s1), dag(s2))
      Op[s1' => 1, s2' => 2, s1 => 1, s2 => 2] = +3 / 2
      Op[s1' => 2, s2' => 1, s1 => 2, s2 => 2] = +1 / 2
      Op[s1' => 3, s2' => 3, s1 => 3, s2 => 4] = -1 / 2
      Op[s1' => 4, s2' => 1, s1 => 4, s2 => 2] = -3 / 2
      return Op
    end

    function ITensors.op(
      ::OpName"β", ::SiteType"_Custom1", ::SiteType"_Custom2", s1::Index, s2::Index
    )
      Op = emptyITensor(s1', s2', dag(s1), dag(s2))
      Op[s1' => 1, s2' => 2, s1 => 1, s2 => 2] = +5 / 2
      Op[s1' => 2, s2' => 1, s1 => 2, s2 => 2] = +3 / 2
      Op[s1' => 3, s2' => 3, s1 => 3, s2 => 4] = -3 / 2
      Op[s1' => 4, s2' => 1, s1 => 4, s2 => 2] = -5 / 2
      return Op
    end

    s = Index(4, "_Custom_, __x")
    Sz = op("Sz", s)
    @test Sz[s' => 1, s => 1] ≈ +3 / 2
    @test Sz[s' => 2, s => 2] ≈ +1 / 2
    @test Sz[s' => 3, s => 3] ≈ -1 / 2
    @test Sz[s' => 4, s => 4] ≈ -3 / 2

    t = Index(4, "_Custom_, __x")
    α = op("α", s, t)
    @test α[s' => 1, t' => 2, s => 1, t => 2] ≈ +3 / 2
    @test α[s' => 2, t' => 1, s => 2, t => 2] ≈ +1 / 2
    @test α[s' => 3, t' => 3, s => 3, t => 4] ≈ -1 / 2
    @test α[s' => 4, t' => 1, s => 4, t => 2] ≈ -3 / 2

    s1 = Index(4, "_Custom1, __x")
    @test_throws ArgumentError op("α", s, s1)

    s2 = Index(4, "_Custom2, __x")
    β = op("β", s1, s2)
    @test β[s1' => 1, s2' => 2, s1 => 1, s2 => 2] ≈ +5 / 2
    @test β[s1' => 2, s2' => 1, s1 => 2, s2 => 2] ≈ +3 / 2
    @test β[s1' => 3, s2' => 3, s1 => 3, s2 => 4] ≈ -3 / 2
    @test β[s1' => 4, s2' => 1, s1 => 4, s2 => 2] ≈ -5 / 2
    @test_throws ArgumentError op("β", s2, s1)
  end

  @testset "Custom OpName with long name" begin
    function ITensors.op(::OpName"my_favorite_operator", ::SiteType"S=1/2", s::Index)
      Op = emptyITensor(s', dag(s))
      Op[s' => 1, s => 1] = 0.11
      Op[s' => 1, s => 2] = 0.12
      Op[s' => 2, s => 1] = 0.21
      Op[s' => 2, s => 2] = 0.22
      return Op
    end

    s = Index(2, "S=1/2, Site")
    Sz = op("my_favorite_operator", s)
    @test Sz[s' => 1, s => 1] ≈ 0.11
    @test Sz[s' => 1, s => 2] ≈ 0.12
    @test Sz[s' => 2, s => 1] ≈ 0.21
    @test Sz[s' => 2, s => 2] ≈ 0.22

    @test OpName(:myop) == OpName("myop")
    @test ITensors.name(OpName(:myop)) == :myop
  end

  @testset "op with more than two indices" begin
    ITensors.space(::SiteType"qubit") = 2

    function ITensors.op(::OpName"rand", ::SiteType"qubit", s::Index...)
      return randomITensor(prime.(s)..., dag.(s)...)
    end

    s = siteinds("qubit", 4)
    o = op("rand", s...)
    @test norm(o) > 0
    @test order(o) == 8
    @test hassameinds(o, (prime.(s)..., s...))
  end

  @testset "Custom Qudit/Boson op" begin
    # Overload Qudit, implicitly defined for Boson as well
    function ITensors.op(::OpName"Qudit_op_1", ::SiteType"Qudit", ds::Int...)
      d = prod(ds)
      return [i * j for i in 1:d, j in 1:d]
    end
    function ITensors.op(::OpName"Qudit_op_2", ::SiteType"Qudit", d::Int)
      return [i * j for i in 1:d, j in 1:d]
    end

    # Overload Boson directly
    function ITensors.op(::OpName"Boson_op_1", ::SiteType"Boson", ds::Int...)
      d = prod(ds)
      return [i * j for i in 1:d, j in 1:d]
    end
    function ITensors.op(::OpName"Boson_op_2", ::SiteType"Boson", d::Int)
      return [i * j for i in 1:d, j in 1:d]
    end

    for st in ["Qudit", "Boson"], ot in ["Qudit", "Boson"]
      if st == "Qudit" && ot == "Boson"
        # Qudit site types don't see Boson overloads
        continue
      end
      d = 4
      s = siteinds(st, 2; dim=d)
      o = op("$(ot)_op_1", s, 1)
      @test o ≈ itensor([i * j for i in 1:d, j in 1:d], s[1]', dag(s[1]))

      o = op("$(ot)_op_1", s, 1, 2)
      @test o ≈ itensor(
        [i * j for i in 1:(d^2), j in 1:(d^2)], s[2]', s[1]', dag(s[2]), dag(s[1])
      )

      d = 4
      s = siteinds(st, 2; dim=d)
      o = op("$(ot)_op_2", s, 1)
      @test o ≈ itensor([i * j for i in 1:d, j in 1:d], s[1]', dag(s[1]))
      @test_throws MethodError op("$(ot)_op_2", s, 1, 2)
    end
  end

  @testset "Custom SiteType using op!" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op!(Op::ITensor, ::OpName"Sz", ::SiteType"_Custom_", s::Index)
      Op[s' => 1, s => 1] = +3 / 2
      Op[s' => 2, s => 2] = +1 / 2
      Op[s' => 3, s => 3] = -1 / 2
      return Op[s' => 4, s => 4] = -3 / 2
    end

    function ITensors.op!(
      Op::ITensor, ::OpName"α", ::SiteType"_Custom_", s1::Index, s2::Index
    )
      Op[s1' => 1, s2' => 2, s1 => 1, s2 => 2] = +3 / 2
      Op[s1' => 2, s2' => 1, s1 => 2, s2 => 2] = +1 / 2
      Op[s1' => 3, s2' => 3, s1 => 3, s2 => 4] = -1 / 2
      return Op[s1' => 4, s2' => 1, s1 => 4, s2 => 2] = -3 / 2
    end

    function ITensors.op!(
      Op::ITensor,
      ::OpName"β",
      ::SiteType"_Custom1",
      ::SiteType"_Custom2",
      s1::Index,
      s2::Index,
    )
      Op[s1' => 1, s2' => 2, s1 => 1, s2 => 2] = +5 / 2
      Op[s1' => 2, s2' => 1, s1 => 2, s2 => 2] = +3 / 2
      Op[s1' => 3, s2' => 3, s1 => 3, s2 => 4] = -3 / 2
      return Op[s1' => 4, s2' => 1, s1 => 4, s2 => 2] = -5 / 2
    end

    s = Index(4, "_Custom_, __x")
    Sz = op("Sz", s)
    @test Sz[s' => 1, s => 1] ≈ +3 / 2
    @test Sz[s' => 2, s => 2] ≈ +1 / 2
    @test Sz[s' => 3, s => 3] ≈ -1 / 2
    @test Sz[s' => 4, s => 4] ≈ -3 / 2

    t = Index(4, "_Custom_, __x")
    α = op("α", s, t)
    @test α[s' => 1, t' => 2, s => 1, t => 2] ≈ +3 / 2
    @test α[s' => 2, t' => 1, s => 2, t => 2] ≈ +1 / 2
    @test α[s' => 3, t' => 3, s => 3, t => 4] ≈ -1 / 2
    @test α[s' => 4, t' => 1, s => 4, t => 2] ≈ -3 / 2

    s1 = Index(4, "_Custom1, __x")
    @test_throws ArgumentError op("α", t, s1)

    s2 = Index(4, "_Custom2, __x")
    β = op("β", s1, s2)
    @test β[s1' => 1, s2' => 2, s1 => 1, s2 => 2] ≈ +5 / 2
    @test β[s1' => 2, s2' => 1, s1 => 2, s2 => 2] ≈ +3 / 2
    @test β[s1' => 3, s2' => 3, s1 => 3, s2 => 4] ≈ -3 / 2
    @test β[s1' => 4, s2' => 1, s1 => 4, s2 => 2] ≈ -5 / 2
    @test_throws ArgumentError op("β", s2, s1)
  end

  @testset "Custom SiteType using older op interface" begin
    # Use "_Custom_" tag even though this example
    # is for S=3/2, because we might define the 
    # "S=3/2" TagType inside ITensors.jl later
    function ITensors.op(::SiteType"_Custom_", s::Index, opname::AbstractString)
      Op = emptyITensor(s', dag(s))
      if opname == "S+"
        Op[s' => 1, s => 2] = sqrt(3)
        Op[s' => 2, s => 3] = 2
        Op[s' => 3, s => 4] = sqrt(3)
      else
        error("Name $opname not recognized for tag \"Custom\"")
      end
      return Op
    end

    s = Index(4, "_Custom_")
    Sp = op("S+", s)
    @test Sp[s' => 1, s => 2] ≈ sqrt(3)
    @test Sp[s' => 2, s => 3] ≈ 2
    @test Sp[s' => 3, s => 4] ≈ sqrt(3)
  end

  @testset "siteind defined by space overload" begin
    ITensors.space(::SiteType"Test1") = 4
    s = siteind("Test1", 3)
    @test dim(s) == 4
    @test hastags(s, "Site,Test1,n=3")

    s = siteind("Test1")
    @test dim(s) == 4
    @test hastags(s, "Site,Test1")
  end

  @testset "siteind defined by siteind overload" begin
    ITensors.siteind(::SiteType"Test2") = Index(4, "Test2")
    s = siteind("Test2", 3)
    @test dim(s) == 4
    @test hastags(s, "Test2,n=3")
  end

  @testset "siteind defined by space overload with QN" begin
    function ITensors.space(::SiteType"Test3")
      return [QN("T", 0) => 2, QN("T", 1) => 1, QN("T", 2) => 1]
    end
    s = siteind("Test3", 3)
    @test dim(s) == 4
    @test hasqns(s)
    @test hastags(s, "Site,Test3,n=3")
  end

  @testset "siteinds defined by space overload" begin
    function ITensors.space(::SiteType"Test4"; conserve_qns=false)
      if conserve_qns
        return [QN("T", 0) => 2, QN("T", 1) => 1, QN("T", 2) => 1]
      end
      return 4
    end

    # Without QNs
    s = siteinds("Test4", 6)
    @test length(s) == 6
    @test dim(s[1]) == 4
    for n in 1:length(s)
      @test hastags(s[n], "Site,Test4,n=$n")
      @test !hasqns(s[n])
    end

    # With QNs
    s = siteinds("Test4", 6; conserve_qns=true)
    @test length(s) == 6
    @test dim(s[1]) == 4
    for n in 1:length(s)
      @test hastags(s[n], "Site,Test4,n=$n")
      @test hasqns(s[n])
    end
  end

  @testset "siteinds defined by siteinds overload" begin
    function ITensors.siteinds(::SiteType"Test5", N; kwargs...)
      return [Index(4, "Test5,n=$n") for n in 1:N]
    end
    s = siteinds("Test5", 8)
    @test length(s) == 8
    @test dim(s[1]) == 4
    for n in 1:length(s)
      @test hastags(s[n], "Test5,n=$n")
    end
  end

  @testset "Version of siteinds taking function argument" begin
    N = 10
    s = siteinds(n -> (n == 1 || n == N) ? "S=1/2" : "S=1", N)
    for n in (1, N)
      @test dim(s[n]) == 2
      @test hastags(s[n], "Site,S=1/2,n=$n")
    end
    for n in 2:(N - 1)
      @test dim(s[n]) == 3
      @test hastags(s[n], "Site,S=1,n=$n")
    end
  end

  @testset "siteinds addtags keyword argument" begin
    N = 4
    s = siteinds("S=1/2", N; addtags="T")
    for n in 1:N
      @test hastags(s[n], "Site,S=1/2,n=$n,T")
    end
  end

  @testset "Error for undefined tag in siteinds,space system" begin
    @test_throws ErrorException siteinds("Missing", 10)
    @test_throws ErrorException siteind("Missing", 3)
    @test isnothing(siteind("Missing"))
  end

  @testset "Various ops input types" begin
    s = siteinds("S=1/2", 4)

    # Vector{Tuple{String,Int}} input
    oa = ops(s, [("Sz", n) for n in 1:length(s)])
    @test length(oa) == length(s)
    @test norm(oa[2] - op("Sz", s, 2)) < 1E-8

    # Vector{Tuple} input
    oa = ops(s, Tuple[("Sz", n) for n in 1:length(s)])
    @test length(oa) == length(s)
    @test norm(oa[2] - op("Sz", s, 2)) < 1E-8
  end

  @testset "Index Values From Strings" begin
    @testset "Val function" begin
      s = siteind("Electron")
      @test val(s, "0") == 1
      @test val(s, "Up") == 2
      @test val(s, "Dn") == 3
      @test val(s, "UpDn") == 4
    end

    @testset "Strings in ITensor get and set" begin
      s = siteind("S=1"; conserve_qns=true)
      T = ITensor(s', dag(s))
      T[s' => "Up", s => "Up"] = +1.0
      T[s' => "Z0", s => "Z0"] = +2.0
      T[s' => "Dn", s => "Dn"] = -1.0
      @test T[1, 1] ≈ +1.0
      @test T[2, 2] ≈ +2.0
      @test T[3, 3] ≈ -1.0

      o = onehot(s => "Z0")
      @test vector(o) ≈ [0, 1, 0]
    end
  end

  @testset "state with variable dimension" begin
    ITensors.space(::SiteType"MyQudit"; dim=2) = dim

    function ITensors.state(::StateName{N}, ::SiteType"MyQudit", s::Index) where {N}
      n = parse(Int, String(N))
      st = zeros(dim(s))
      st[n + 1] = 1.0
      return itensor(st, s)
    end

    s = siteind("MyQudit"; dim=3)
    v0 = state(s, "0")
    v1 = state(s, "1")
    v2 = state(s, "2")
    @test v0 == state("0", s)
    @test v1 == state("1", s)
    @test v2 == state("2", s)
    @test dim(v0) == 3
    @test dim(v1) == 3
    @test dim(v2) == 3
    @test v0[s => 1] == 1
    @test v0[s => 2] == 0
    @test v0[s => 3] == 0
    @test v1[s => 1] == 0
    @test v1[s => 2] == 1
    @test v1[s => 3] == 0
    @test v2[s => 1] == 0
    @test v2[s => 2] == 0
    @test v2[s => 3] == 1
    @test_throws BoundsError state(s, "3")
  end

  @testset "state with parameters" begin
    ITensors.state(::StateName"phase", ::SiteType"Qubit"; θ::Real) = [cos(θ), sin(θ)]
    s = siteind("Qubit")
    @test state("phase", s; θ=π / 6) ≈ itensor([cos(π / 6), sin(π / 6)], s)
  end

  @testset "state with variable dimension (deprecated)" begin
    ITensors.space(::SiteType"MyQudit2"; dim=2) = dim

    # XXX: This syntax is deprecated, only testing for
    # backwards compatibility. Should return the
    # ITensor `itensor(st, s)`.
    function ITensors.state(::StateName{N}, ::SiteType"MyQudit2", s::Index) where {N}
      n = parse(Int, String(N))
      st = zeros(dim(s))
      st[n + 1] = 1.0
      return st
    end

    s = siteind("MyQudit2"; dim=3)
    v0 = state(s, "0")
    v1 = state(s, "1")
    v2 = state(s, "2")
    @test v0 == state("0", s)
    @test v1 == state("1", s)
    @test v2 == state("2", s)
    @test dim(v0) == 3
    @test dim(v1) == 3
    @test dim(v2) == 3
    @test v0[s => 1] == 1
    @test v0[s => 2] == 0
    @test v0[s => 3] == 0
    @test v1[s => 1] == 0
    @test v1[s => 2] == 1
    @test v1[s => 3] == 0
    @test v2[s => 1] == 0
    @test v2[s => 2] == 0
    @test v2[s => 3] == 1
    @test_throws BoundsError state(s, "3")
  end

  @testset "StateName methods" begin
    @test StateName(ITensors.SmallString("a")) == StateName("a")
    @test ITensors.name(StateName("a")) == ITensors.SmallString("a")
  end

  @testset "Regression test for state overload" begin
    ITensors.space(::SiteType"Xev") = 8
    function ITensors.state(::StateName"0", ::SiteType"Xev")
      return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end
    s = siteind("Xev")
    @test state(s, "0") ≈ ITensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], s)
  end

  @testset "function applied to a gate" begin
    s = siteinds("Qubit", 2)

    θ = 0.1
    rx = array(op("Rx", s[1]; θ=0.1))
    exp_rx = exp(rx)
    gtest = op(x -> exp(x), "Rx", s[1]; θ=0.1)
    @test exp_rx ≈ array(op(x -> exp(x), "Rx", s[1]; θ=0.1))
    @test exp_rx ≈ array(op(x -> exp(x), ("Rx", 1, (θ=0.1,)), s))

    cx = 0.1 * reshape(array(op("CX", s[1], s[2])), (4, 4))
    exp_cx = reshape(exp(cx), (2, 2, 2, 2))
    @test exp_cx ≈ array(op(x -> exp(0.1 * x), "CX", s[1], s[2]))
    @test exp_cx ≈ array(op(x -> exp(0.1 * x), ("CX", (1, 2)), s))
  end

  @testset "Haar-random unitary RandomUnitary" begin
    s = siteinds(2, 3)

    U = op("RandomUnitary", s, 1, 2)
    @test eltype(U) == ComplexF64
    @test order(U) == 4
    @test is_unitary(U; rtol=1e-15)

    U = op("RandomUnitary", s, 1, 2, 3)
    @test eltype(U) == ComplexF64
    @test order(U) == 6
    @test is_unitary(U; rtol=1e-15)

    U = op("RandomUnitary", s, 1, 2; eltype=Float64)
    @test eltype(U) == Float64
    @test order(U) == 4
    @test is_unitary(U; rtol=1e-15)

    U = op("RandomUnitary", s, 1, 2, 3; eltype=Float64)
    @test eltype(U) == Float64
    @test order(U) == 6
    @test is_unitary(U; rtol=1e-15)
  end
end

nothing
