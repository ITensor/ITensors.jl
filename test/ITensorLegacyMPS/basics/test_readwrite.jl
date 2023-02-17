using ITensors, HDF5, Test

include(joinpath(@__DIR__, "utils", "util.jl"))

@testset "HDF5 Read and Write" begin
  @testset "MPO/MPS" begin
    N = 6
    sites = siteinds("S=1/2", N)

    # MPO
    mpo = makeRandomMPO(sites)

    h5open("data.h5", "w") do fo
      write(fo, "mpo", mpo)
    end

    h5open("data.h5", "r") do fi
      rmpo = read(fi, "mpo", MPO)
      @test prod([norm(rmpo[i] - mpo[i]) / norm(mpo[i]) < 1E-10 for i in 1:N])
    end

    # MPS
    mps = makeRandomMPS(sites)
    h5open("data.h5", "w") do fo
      write(fo, "mps", mps)
    end

    h5open("data.h5", "r") do fi
      rmps = read(fi, "mps", MPS)
      @test prod([norm(rmps[i] - mps[i]) / norm(mps[i]) < 1E-10 for i in 1:N])
    end
  end

  #
  # Clean up the test hdf5 file
  #
  rm("data.h5"; force=true)
end

nothing
