using ITensors, HDF5, Test

include(joinpath(@__DIR__, "utils", "util.jl"))

@testset "HDF5 Read and Write" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")

  @testset "TagSet" begin
    ts = TagSet("A,Site,n=2")
    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "tags", ts)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rts = read(fi, "tags", TagSet)
      @test rts == ts
    end
  end

  @testset "Index" begin
    i = Index(3, "Site,S=1")
    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "index", i)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      ri = read(fi, "index", Index)
      @test ri == i
    end

    # primed Index
    i = Index(3, "Site,S=1")
    i = prime(i, 2)
    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "index", i)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      ri = read(fi, "index", Index)
      @test ri == i
    end
  end

  @testset "IndexSet" begin
    is = IndexSet(i, j, k)

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "inds", is)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      ris = read(fi, "inds", IndexSet)
      @test ris == is
    end
  end

  @testset "Dense ITensor" begin

    # default constructed case
    T = ITensor()

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "defaultT", T)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rT = read(fi, "defaultT", ITensor)
      @test typeof(storage(T)) == typeof(storage(ITensor()))
    end

    # real case
    T = randomITensor(i, j, k)

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "T", T)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rT = read(fi, "T", ITensor)
      @test norm(rT - T) / norm(T) < 1E-10
    end

    # complex case
    T = randomITensor(ComplexF64, i, j, k)

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "complexT", T)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rT = read(fi, "complexT", ITensor)
      @test norm(rT - T) / norm(T) < 1E-10
    end
  end

  @testset "Delta ITensor" begin
    #
    # Delta ITensor
    #
    Δ = δ(i, i')
    cΔ = δ(ComplexF64, i, i')
    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      fo["delta_tensor"] = Δ
      fo["c_delta_tensor"] = cΔ
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rΔ = read(fi, "delta_tensor", ITensor)
      rcΔ = read(fi, "c_delta_tensor", ITensor)
      @test rΔ ≈ Δ
      @test rcΔ ≈ cΔ
    end
  end
  @testset "Diag ITensor" begin

    #
    # Diag ITensor
    #
    dk = dim(k)
    D = diagITensor(randn(dk), k, k')
    C = diagITensor(randn(ComplexF64, dk), k, k')
    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      fo["diag_tensor"] = D
      fo["c_diag_tensor"] = C
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rD = read(fi, "diag_tensor", ITensor)
      rC = read(fi, "c_diag_tensor", ITensor)
      @test rD ≈ D
      @test rC ≈ C
    end
  end

  @testset "QN ITensor" begin
    i = Index(QN("A", -1) => 3, QN("A", 0) => 4, QN("A", +1) => 3; tags="i")
    j = Index(QN("A", -2) => 2, QN("A", 0) => 3, QN("A", +2) => 2; tags="j")
    k = Index(QN("A", -1) => 1, QN("A", 0) => 1, QN("A", +1) => 1; tags="k")

    # real case
    T = randomITensor(QN("A", 1), i, j, k)

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "T", T)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rT = read(fi, "T", ITensor)
      @test rT ≈ T
    end

    # complex case
    T = randomITensor(ComplexF64, i, j, k)

    h5open(joinpath(@__DIR__, "data.h5"), "w") do fo
      write(fo, "complexT", T)
    end

    h5open(joinpath(@__DIR__, "data.h5"), "r") do fi
      rT = read(fi, "complexT", ITensor)
      @test rT ≈ T
    end
  end

  @testset "DownwardCompat" begin
    h5open(joinpath(@__DIR__, "utils", "testfilev0.1.41.h5"), "r") do fi
      ITensorName = "ITensorv0.1.41"

      # ITensor version <= v0.1.41 uses the `store` key for ITensor data storage
      # whereas v >= 0.2 uses `storage` as key
      @test haskey(read(fi, ITensorName), "store")
      @test read(fi, ITensorName, ITensor) isa ITensor
    end
  end

  #
  # Clean up the test hdf5 file
  #
  rm(joinpath(@__DIR__, "data.h5"); force=true)
end
