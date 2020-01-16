
using ITensors,
      HDF5,
      Test

@testset "HDF5 Read and Write" begin

  i = Index(2,"i")
  j = Index(3,"j")
  k = Index(4,"k")

  @testset "TagSet" begin
    ts = TagSet("A,Site,n=2")
    fo = h5open("data.h5","w")
    write(fo,"tags",ts)
    close(fo)

    fi = h5open("data.h5","r")
    rts = read(fi,"tags",TagSet)
    close(fi)
    @test rts == ts
  end

  @testset "Index" begin
    i = Index(3,"Site,S=1")
    fo = h5open("data.h5","w")
    write(fo,"index",i)
    close(fo)

    fi = h5open("data.h5","r")
    ri = read(fi,"index",Index)
    close(fi)
    @test ri == i

    # primed Index
    i = Index(3,"Site,S=1")
    i = prime(i,2)
    fo = h5open("data.h5","w")
    write(fo,"index",i)
    close(fo)

    fi = h5open("data.h5","r")
    ri = read(fi,"index",Index)
    close(fi)
    @test ri == i
  end

  @testset "IndexSet" begin
    is = IndexSet(i,j,k)

    fo = h5open("data.h5","w")
    write(fo,"inds",is)
    close(fo)

    fi = h5open("data.h5","r")
    ris = read(fi,"inds",IndexSet)
    close(fi)
    @test ris == is
  end

  @testset "ITensor" begin

    # default constructed case
    T = ITensor()

    fo = h5open("data.h5","w")
    write(fo,"defaultT",T)
    close(fo)

    fi = h5open("data.h5","r")
    rT = read(fi,"defaultT",ITensor)
    close(fi)
    @test typeof(store(T)) == typeof(store(ITensor()))

    # real case
    T = randomITensor(i,j,k)

    fo = h5open("data.h5","w")
    write(fo,"T",T)
    close(fo)

    fi = h5open("data.h5","r")
    rT = read(fi,"T",ITensor)
    close(fi)
    @test norm(rT-T)/norm(T) < 1E-10

    # complex case
    T = randomITensor(ComplexF64,i,j,k)

    fo = h5open("data.h5","w")
    write(fo,"complexT",T)
    close(fo)

    fi = h5open("data.h5","r")
    rT = read(fi,"complexT",ITensor)
    close(fi)
    @test norm(rT-T)/norm(T) < 1E-10
  end

  #
  # Clean up the test hdf5 file
  #
  rm("data.h5",force=true)

end
