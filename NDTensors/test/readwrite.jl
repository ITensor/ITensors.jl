using NDTensors, Test
using HDF5

@testset "Write to Disk and Read from Disk" begin
  @testset "HDF5 readwrite Dense storage" begin
    # Real case

    D = randomTensor(3, 4)

    fo = h5open("data.h5", "w")
    write(fo, "D", D.store)
    close(fo)

    fi = h5open("data.h5", "r")
    rDstore = read(fi, "D", Dense{Float64})
    close(fi)
    @test rDstore ≈ D.store

    # Complex case

    D = randomTensor(ComplexF64, 3, 4)

    fo = h5open("data.h5", "w")
    write(fo, "D", D.store)
    close(fo)

    fi = h5open("data.h5", "r")
    rDstore = read(fi, "D", Dense{ComplexF64})
    close(fi)
    @test rDstore ≈ D.store
  end

  @testset "HDF5 readwrite BlockSparse storage" begin
    # Indices
    indsA = ([2, 3], [4, 5])

    # Locations of non-zero blocks
    locs = [(1, 2), (2, 1)]

    # Real case

    B = randomBlockSparseTensor(locs, indsA)

    fo = h5open("data.h5", "w")
    write(fo, "B", B.store)
    close(fo)

    fi = h5open("data.h5", "r")
    rBstore = read(fi, "B", BlockSparse{Float64})
    close(fi)
    @test rBstore ≈ B.store

    # Complex case

    B = randomBlockSparseTensor(ComplexF64, locs, indsA)

    fo = h5open("data.h5", "w")
    write(fo, "B", B.store)
    close(fo)

    fi = h5open("data.h5", "r")
    rBstore = read(fi, "B", BlockSparse{ComplexF64})
    close(fi)
    @test rBstore ≈ B.store
  end

  #
  # Clean up the test hdf5 file
  #
  rm("data.h5"; force=true)
end

nothing
