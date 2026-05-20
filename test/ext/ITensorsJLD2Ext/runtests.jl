@eval module $(gensym())
using ITensors: ITensors, ITensor, Index, QN, TagSet, dag, delta, dim, hasind, hassameinds,
    inds, random_itensor
using JLD2: jldsave, load
using Test: @test, @testset

@testset "JLD2 round-trip: QN" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        q = QN("Sz", 1)
        jldsave(fname; q)
        @test load(fname, "q") == q

        q2 = QN(("Sz", 1), ("Nf", 2, 2))
        jldsave(fname; q = q2)
        @test load(fname, "q") == q2

        q0 = QN()
        jldsave(fname; q = q0)
        @test load(fname, "q") == q0
    end
end

@testset "JLD2 round-trip: TagSet" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        for ts in (TagSet("a,b,c"), TagSet(""), TagSet("Site"))
            jldsave(fname; ts)
            @test load(fname, "ts") == ts
        end
    end
end

@testset "JLD2 round-trip: Index" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        i = Index(4, "Site,n=1")
        jldsave(fname; i)
        @test load(fname, "i") == i

        i2p = Index(3; tags = "Link", dir = ITensors.In)'
        jldsave(fname; i = i2p)
        @test load(fname, "i") == i2p

        iq = Index([QN("Sz", -1) => 1, QN("Sz", 0) => 2, QN("Sz", 1) => 1], "Site,n=1")
        jldsave(fname; i = iq)
        @test load(fname, "i") == iq
    end
end

@testset "JLD2 round-trip: Dense ITensor" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        i, j = Index(3, "i"), Index(4, "j")
        A = random_itensor(Float64, i, j)
        jldsave(fname; A)
        A_loaded = load(fname, "A")
        @test A_loaded ≈ A
        @test inds(A_loaded) == inds(A)

        Ac = random_itensor(ComplexF64, i, j)
        jldsave(fname; A = Ac)
        @test load(fname, "A") ≈ Ac
    end
end

@testset "JLD2 round-trip: BlockSparse ITensor" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        i = Index([QN("Sz", -1) => 1, QN("Sz", 0) => 2, QN("Sz", 1) => 1], "i")
        j = Index([QN("Sz", -1) => 1, QN("Sz", 0) => 2, QN("Sz", 1) => 1], "j")
        A = random_itensor(Float64, i, dag(j))
        jldsave(fname; A)
        A_loaded = load(fname, "A")
        @test A_loaded ≈ A
        @test inds(A_loaded) == inds(A)
    end
end

@testset "JLD2 round-trip: Diag ITensor" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        i, j = Index(3, "i"), Index(3, "j")
        A = delta(Float64, i, j)
        jldsave(fname; A)
        A_loaded = load(fname, "A")
        @test A_loaded ≈ A
        @test inds(A_loaded) == inds(A)
    end
end

end # module
