@eval module $(gensym())
using JLD2: jldsave, load
using NDTensors:
    NDTensors, Block, BlockOffsets, BlockSparse, Dense, Diag, DiagBlockSparse, EmptyStorage
using Test: @test, @testset

@testset "JLD2 round-trip: Dense storage" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        d = Dense(randn(12))
        jldsave(fname; d)
        d_loaded = load(fname, "d")
        @test d_loaded isa Dense{Float64}
        @test NDTensors.data(d_loaded) == NDTensors.data(d)

        dc = Dense(randn(ComplexF64, 6))
        jldsave(fname; d = dc)
        dc_loaded = load(fname, "d")
        @test dc_loaded isa Dense{ComplexF64}
        @test NDTensors.data(dc_loaded) == NDTensors.data(dc)
    end
end

@testset "JLD2 round-trip: BlockSparse storage" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        boffs = BlockOffsets{2}()
        insert!(boffs, Block(UInt(1), UInt(1)), 0)
        insert!(boffs, Block(UInt(2), UInt(2)), 4)
        bs = BlockSparse(randn(8), boffs)
        jldsave(fname; bs)
        bs_loaded = load(fname, "bs")
        @test bs_loaded isa BlockSparse{Float64}
        @test NDTensors.data(bs_loaded) == NDTensors.data(bs)
        @test NDTensors.blockoffsets(bs_loaded) == NDTensors.blockoffsets(bs)
    end
end

@testset "JLD2 round-trip: Diag storage (nonuniform)" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        d = Diag(randn(5))
        jldsave(fname; d)
        d_loaded = load(fname, "d")
        @test d_loaded isa Diag{Float64, Vector{Float64}}
        @test NDTensors.data(d_loaded) == NDTensors.data(d)
    end
end

@testset "JLD2 round-trip: Diag storage (uniform)" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        d = Diag(3.14)
        jldsave(fname; d)
        d_loaded = load(fname, "d")
        @test d_loaded isa Diag{Float64, Float64}
        @test NDTensors.data(d_loaded) == NDTensors.data(d)
    end
end

@testset "JLD2 round-trip: DiagBlockSparse storage (nonuniform)" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        boffs = BlockOffsets{2}()
        insert!(boffs, Block(UInt(1), UInt(1)), 0)
        insert!(boffs, Block(UInt(2), UInt(2)), 1)
        insert!(boffs, Block(UInt(3), UInt(3)), 3)
        dbs = DiagBlockSparse(randn(4), boffs)
        jldsave(fname; dbs)
        dbs_loaded = load(fname, "dbs")
        @test dbs_loaded isa DiagBlockSparse{Float64, Vector{Float64}}
        @test NDTensors.data(dbs_loaded) == NDTensors.data(dbs)
        @test NDTensors.blockoffsets(dbs_loaded) == NDTensors.blockoffsets(dbs)
    end
end

@testset "JLD2 round-trip: DiagBlockSparse storage (uniform)" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        boffs = BlockOffsets{2}()
        insert!(boffs, Block(UInt(1), UInt(1)), 0)
        insert!(boffs, Block(UInt(2), UInt(2)), 1)
        insert!(boffs, Block(UInt(3), UInt(3)), 3)
        dbs = DiagBlockSparse(1.0, boffs)
        jldsave(fname; dbs)
        dbs_loaded = load(fname, "dbs")
        @test dbs_loaded isa DiagBlockSparse{Float64, Float64}
        @test NDTensors.data(dbs_loaded) == NDTensors.data(dbs)
        @test NDTensors.blockoffsets(dbs_loaded) == NDTensors.blockoffsets(dbs)
    end
end

@testset "JLD2 round-trip: EmptyStorage" begin
    mktempdir() do dir
        fname = joinpath(dir, "test.jld2")

        es = NDTensors.EmptyStorage(Float64)
        jldsave(fname; es)
        es_loaded = load(fname, "es")
        @test es_loaded isa EmptyStorage
        @test eltype(es_loaded) == Float64
    end
end

end # module
