
using ITensors,
      HDF5,
      Test

@testset "HDF5 Read and Write" begin

  @testset "TagSet" begin
    ts = TagSet("A,Site,n=2")
    fo = h5open("data.h5","w")
    write(fo,ts)
    close(fo)

    fi = h5open("data.h5","r")
    rts = read(fi,TagSet)
    close(fi)
    @test rts == ts
  end

  @testset "Index" begin
    i = Index(3,"Site,S=1")
    fo = h5open("data.h5","w")
    write(fo,i)
    close(fo)

    fi = h5open("data.h5","r")
    ri = read(fi,Index)
    close(fi)
    @test ri == i
  end

  @testset "IndexSet" begin
    i = Index(2,"i")
    j = Index(3,"j")
    k = Index(4,"k")
    is = IndexSet(i,j,k)

    fo = h5open("data.h5","w")
    write(fo,is)
    close(fo)

    fi = h5open("data.h5","r")
    ris = read(fi,IndexSet)
    close(fi)
    @test ris == is
  end

end
