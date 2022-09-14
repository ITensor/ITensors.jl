using ITensors, HDF5, Test

include("util.jl")

@testset "HDF5 Read and Write" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")

  @testset "TagSet" begin
    ts = TagSet("A,Site,n=2")
    fo = h5open("data.h5", "w")
    write(fo, "tags", ts)
    close(fo)

    fi = h5open("data.h5", "r")
    rts = read(fi, "tags", TagSet)
    close(fi)
    @test rts == ts
  end

  @testset "Index" begin
    i = Index(3, "Site,S=1")
    fo = h5open("data.h5", "w")
    write(fo, "index", i)
    close(fo)

    fi = h5open("data.h5", "r")
    ri = read(fi, "index", Index)
    close(fi)
    @test ri == i

    # primed Index
    i = Index(3, "Site,S=1")
    i = prime(i, 2)
    fo = h5open("data.h5", "w")
    write(fo, "index", i)
    close(fo)

    fi = h5open("data.h5", "r")
    ri = read(fi, "index", Index)
    close(fi)
    @test ri == i
  end

  @testset "IndexSet" begin
    is = IndexSet(i, j, k)

    fo = h5open("data.h5", "w")
    write(fo, "inds", is)
    close(fo)

    fi = h5open("data.h5", "r")
    ris = read(fi, "inds", IndexSet)
    close(fi)
    @test ris == is
  end

  @testset "Dense ITensor" begin

    # default constructed case
    T = ITensor()

    fo = h5open("data.h5", "w")
    write(fo, "defaultT", T)
    close(fo)

    fi = h5open("data.h5", "r")
    rT = read(fi, "defaultT", ITensor)
    close(fi)
    @test typeof(storage(T)) == typeof(storage(ITensor()))

    # real case
    T = randomITensor(i, j, k)

    fo = h5open("data.h5", "w")
    write(fo, "T", T)
    close(fo)

    fi = h5open("data.h5", "r")
    rT = read(fi, "T", ITensor)
    close(fi)
    @test norm(rT - T) / norm(T) < 1E-10

    # complex case
    T = randomITensor(ComplexF64, i, j, k)

    fo = h5open("data.h5", "w")
    write(fo, "complexT", T)
    close(fo)

    fi = h5open("data.h5", "r")
    rT = read(fi, "complexT", ITensor)
    close(fi)
    @test norm(rT - T) / norm(T) < 1E-10
  end

  @testset "Diag ITensor" begin
    #
    # Delta ITensor
    #
    fo = h5open("data.h5", "w")
    Δ = δ(i, i')
    cΔ = δ(ComplexF64, i, i')
    fo["delta_tensor"] = Δ
    fo["c_delta_tensor"] = cΔ
    close(fo)

    fi = h5open("data.h5", "r")
    rΔ = read(fi, "delta_tensor", ITensor)
    rcΔ = read(fi, "c_delta_tensor", ITensor)
    close(fi)
    @test rΔ ≈ Δ
    @test rcΔ ≈ cΔ

    #
    # Diag ITensor
    #
    fo = h5open("data.h5", "w")
    dk = dim(k)
    D = diagITensor(randn(dk), k, k')
    C = diagITensor(randn(ComplexF64, dk), k, k')
    fo["diag_tensor"] = D
    fo["c_diag_tensor"] = C
    close(fo)

    fi = h5open("data.h5", "r")
    rD = read(fi, "diag_tensor", ITensor)
    rC = read(fi, "c_diag_tensor", ITensor)
    close(fi)
    @test rD ≈ D
    @test rC ≈ C
  end

  @testset "QN ITensor" begin
    i = Index(QN("A", -1) => 3, QN("A", 0) => 4, QN("A", +1) => 3; tags="i")
    j = Index(QN("A", -2) => 2, QN("A", 0) => 3, QN("A", +2) => 2; tags="j")
    k = Index(QN("A", -1) => 1, QN("A", 0) => 1, QN("A", +1) => 1; tags="k")

    # real case
    T = randomITensor(QN("A", 1), i, j, k)

    fo = h5open("data.h5", "w")
    write(fo, "T", T)
    close(fo)

    fi = h5open("data.h5", "r")
    rT = read(fi, "T", ITensor)
    close(fi)
    @test rT ≈ T

    # complex case
    T = randomITensor(ComplexF64, i, j, k)

    fo = h5open("data.h5", "w")
    write(fo, "complexT", T)
    close(fo)

    fi = h5open("data.h5", "r")
    rT = read(fi, "complexT", ITensor)
    close(fi)
    @test rT ≈ T
  end

  @testset "MPO/MPS" begin
    N = 6
    sites = siteinds("S=1/2", N)

    # MPO
    mpo = makeRandomMPO(sites)

    fo = h5open("data.h5", "w")
    write(fo, "mpo", mpo)
    close(fo)

    fi = h5open("data.h5", "r")
    rmpo = read(fi, "mpo", MPO)
    close(fi)
    @test prod([norm(rmpo[i] - mpo[i]) / norm(mpo[i]) < 1E-10 for i in 1:N])

    # MPS
    mps = makeRandomMPS(sites)
    fo = h5open("data.h5", "w")
    write(fo, "mps", mps)
    close(fo)

    fi = h5open("data.h5", "r")
    rmps = read(fi, "mps", MPS)
    close(fi)
    @test prod([norm(rmps[i] - mps[i]) / norm(mps[i]) < 1E-10 for i in 1:N])
  end

  @testset "DownwardCompat" begin
    fi = h5open("testfilev0.1.41.h5", "r")

    ITensorName = "ITensorv0.1.41"

    # ITensor version <= v0.1.41 uses the `store` key for ITensor data storage
    # whereas v >= 0.2 uses `storage` as key
    @test haskey(read(fi, ITensorName), "store")
    @test read(fi, ITensorName, ITensor) isa ITensor
    close(fi)
  end

  #
  # Clean up the test hdf5 file
  #
  rm("data.h5"; force=true)
end

nothing
