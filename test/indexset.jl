using ITensors
using Test
using Compat

@testset "IndexSet" begin
  idim = 2
  jdim = 3
  kdim = 4
  ldim = 5
  i = Index(idim,"i")
  j = Index(jdim,"j")
  k = Index(kdim,"k")
  l = Index(ldim,"l")
  @testset "show" begin
    indset = IndexSet(i,j,k)
    @test length(sprint(show, indset)) > 1
  end
  @testset "Basic constructors" begin
    I = IndexSet(i,j,k)
    @test IndexSet(I) === I
    @test l ∈ IndexSet(I..., l) 
    @test l ∈ IndexSet(l, I...)
    @test length(IndexSet{2}(i,j)) == 2
  end
  @testset "length of IndexSet and friends" begin
    @test length(typeof(IndexSet{2}(i,j))) == 2
    @test length(IndexSet(i,j)) == 2
    @test length(typeof(IndexSet(i,j))) == 2
  end
  @testset "Convert to Index" begin
    @test Index(IndexSet(i)) === i
    @test_throws ErrorException Index(IndexSet(i, j))
  end
  @testset "Index dimensions" begin
    I = IndexSet(i,j,k)
    @test dim(I) == idim*jdim*kdim
    @test dims(I) == (idim,jdim,kdim)
    @test dim(I,1) == idim
    @test dim(I,2) == jdim
    @test dim(I,3) == kdim

    @test maxdim(I) == max(idim,jdim,kdim)
  end

  @testset "Set operations" begin
    I1 = IndexSet(i, j, k)
    I2 = IndexSet(k, l)
    I3 = IndexSet(j, l)
    @test hassameinds(I1, (k, j, i))
    @test firstsetdiff(I1, I2, I3) == i
    @test isnothing(firstsetdiff(I1, IndexSet(k, j, i)))
    @test setdiff(I1, I2) == [i, j]
    @test setdiff(Order(2), I1, I2) == IndexSet(i, j)
    @test hassameinds(setdiff(I1, I2), IndexSet(i, j))
    @test hassameinds(setdiff(Order(2), I1, I2), IndexSet(i, j))
    @test hassameinds(setdiff(I1, I2), (j, i))
    @test hassameinds(setdiff(Order(2), I1, I2), (j, i))
    @test I1 ∩ I2 == [k]
    @test hassameinds(I1 ∩ I2, IndexSet(k))
    @test firstintersect(I1, I2) == k
    @test isnothing(firstintersect(I1, IndexSet(l)))
    @test intersect(I1, IndexSet(j, l)) == [j]
    @test intersect(Order(1), I1, IndexSet(j, l)) == IndexSet(j)
    @test hassameinds(intersect(I1, IndexSet(j, l)), IndexSet(j))
    @test hassameinds(intersect(Order(1), I1, IndexSet(j, l)), IndexSet(j))
    @test firstintersect(I1, IndexSet(j, l)) == j
    @test intersect(I1, IndexSet(j, k)) == [j, k]
    @test intersect(Order(2), I1, IndexSet(j, k)) == IndexSet(j, k)
    @test hassameinds(intersect(I1, (j, k)), IndexSet(j, k))
    @test hassameinds(intersect(I1, (j, k, l)), (j, k))
    @test filter(I1, "i") == IndexSet(i)
    @test filter(I1; tags = "i") == IndexSet(i)
    @test filter(I1; inds = j) == IndexSet(j)
    @test filter(I1; tags = "i", inds = j) == IndexSet()
    @test filter(I1; plev = 1, inds = j) == IndexSet()
    @test filter(I1; plev = 0, inds = k) == IndexSet(k)
    @test filter(I1; plev = 0) == IndexSet(i, j, k)
    @test filter(I1; inds = l) == IndexSet()
    @test hassameinds(filter(I1, "i"), IndexSet(i))
    @test getfirst(I1, "j") == j
    @test isnothing(getfirst(I1, "l"))
    @test findfirst(I1, i) == 1
    @test findfirst(I1, j) == 2
    @test findfirst(I1, k) == 3
    @test isnothing(findfirst(I1, Index(2)))
  end

  @testset "Set operations with Order" begin
    i,j,k,l = Index.(2, ("i", "j", "k", "l"))

    Iij = IndexSet(i, j)
    Ijl = IndexSet(j, l)
    Ikl = IndexSet(k, l)
    Iijk = IndexSet(i, j, k)

    #
    # setdiff 
    # intersect
    # symdiff
    # union
    # filter
    #

    #
    # setdiff
    #

    @test setdiff(Iijk, Ikl) == [i, j]
    @test setdiff(Order(2), Iijk, Ikl) == IndexSet(i, j)

    @test setdiff(Iij, Iijk) == Index{Int}[]
    @test setdiff(Order(0), Iij, Iijk) == IndexSet()

    @test setdiff(Iijk, Ikl; tags = "i") == [i]
    @test setdiff(Order(1), Iijk, Ikl; tags = "i") == IndexSet(i)

    @test setdiff(Iijk, Ikl; tags = not("i")) == [j]
    @test setdiff(Order(1), Iijk, Ikl; tags = not("i")) == IndexSet(j)

    @test setdiff(Iijk, Ijl, Ikl) == [i]
    @test setdiff(Order(1), Iijk, Ijl, Ikl) == IndexSet(i)

    #
    # intersect
    #

    @test intersect(Iijk, Ikl) == [k]
    @test intersect(Order(1), Iijk, Ikl) == IndexSet(k)

    @test intersect(Iijk, Iij) == [i, j]
    @test intersect(Order(2), Iijk, Iij) == IndexSet(i, j)

    @test intersect(Iijk, Iij; tags = "i") == [i]
    @test intersect(Order(1), Iijk, Iij; tags = "i") == IndexSet(i)

    #
    # symdiff
    #

    @test symdiff(Iijk, Ikl) == [i, j, l]
    #@test symdiff(Order(3), Iijk, Ikl) == IndexSet(i, j, l)

    @test symdiff(Iijk, Iij) == [k]
    #@test symdiff(Order(3), Iijk, Iij) == IndexSet(i, j, k)

    #@test symdiff(Iijk, Iij; tags = "i") == [i]
    #@test symdiff(Order(1), Iijk, Iij; tags = "i") == IndexSet(i)

    #
    # union
    #

    @test union(Iijk, Ikl) == [i, j, k, l]
    #@test union(Order(4), Iijk, Ikl) == IndexSet(i, j, k, l)

    @test union(Iijk, Iij) == [i, j, k]
    #@test union(Order(3), Iijk, Iij) == IndexSet(i, j, k)

    #@test union(Iijk, Iij; tags = "i") == [i]
    #@test union(Order(1), Iijk, Iij; tags = "i") == IndexSet(i)
  end

  @testset "intersect index ordering" begin
    I = IndexSet(i,k,j)
    J = IndexSet(j,l,i)
    # Test that intersect respects the ordering
    # of the indices in the first IndexSet
    @test hassameinds(intersect(I,J),IndexSet(i,j))
    @test hassameinds(intersect(J,I),IndexSet(j,i))
  end
  @testset "adjoint" begin
    I = IndexSet(i,k,j)
    @test adjoint(I) == IndexSet(i', k', j')
  end
  @testset "mapprime" begin
    I = IndexSet(i',k'',j)
    @test mapprime(I,1,5) == IndexSet(i^5,k'',j)
    @test mapprime(I,2,0) == IndexSet(i',k,j)

    J = IndexSet(i,j,k')
    @test mapprime(J,0,2) == IndexSet(i'',j'',k')

    J = mapprime(J,1,5)
    @test J == IndexSet(i,j,k^5)
  end
  @testset "strides" begin
    I = IndexSet(i, j)
    @test NDTensors.dim_to_strides(I) == (1, idim)
    @test NDTensors.dim_to_stride(I, 1) == 1
    @test NDTensors.dim_to_stride(I, 2) == idim
  end
  @testset "setprime" begin
    I = IndexSet(i, j)
    J = setprime(I, 2, i)
    @test i'' ∈ J
  end
  @testset "prime" begin
    I = IndexSet(i, j)
    J = prime(I, j)
    @test i ∈ J
    @test j' ∈ J
    J = prime(I; inds = j)
    @test i ∈ J
    @test j' ∈ J
    J = prime(I; inds = not(j))
    @test i' ∈ J
    @test j ∈ J
  end
  @testset "noprime" begin
    I = IndexSet(i'', j')
    J = noprime(I)
    @test i ∈ J
    @test j ∈ J
  end
  @testset "swapprime" begin
    I = IndexSet(i,j)
    @test swapprime(I,0,1) == IndexSet(i',j')
    @test swapprime(I,0,4) == IndexSet(i^4,j^4)
    I = IndexSet(i,j'')
    @test swapprime(I,2,0) == IndexSet(i'',j)
    I = IndexSet(i,j'',k,l)
    @test swapprime(I,2,0) == IndexSet(i'',j,k'',l'')
    I = IndexSet(i,k'',j'')
    @test swapprime(I,2,1) == IndexSet(i,k',j')
    # In-place version:
    I = IndexSet(i,k'',j''')
    I = swapprime(I,2,0)
    @test I == IndexSet(i'',k,j''')
    # With tags specified:
    I = IndexSet(i,k,j)
    @test swapprime(I,0,1,"i") == IndexSet(i',k,j)
    @test swapprime(I,0,1,"j") == IndexSet(i,k,j')
      
    I = IndexSet(i,i',j)
    @test swapprime(I,0,1,"i") == IndexSet(i',i,j)
    @test swapprime(I,0,1,"j") == IndexSet(i,i',j')
  end

  @testset "swaptags" begin
    i1 = Index(2,"Site,A")
    i2 = Index(2,"Site,B")
    is = IndexSet(i1,i2)
    sis = swaptags(is,"Site","Link")
    for j in sis
      @test !hastags(j,"Site")
      @test hastags(j,"Link")
    end
  end

  @testset "hastags" begin
    i = Index(2, "i, x")
    j = Index(2, "j, x")
    is = IndexSet(i, j)
    @test hastags(is, "i")
    @test anyhastags(is, "i")
    @test !allhastags(is, "i")
    @test allhastags(is, "x")
  end

  @testset "broadcasting" begin
    I = IndexSet(i, j)
    J = prime.(I)
    @test J isa IndexSet
    @test i' ∈ J
    @test j' ∈ J
  end
end

nothing
