using ITensors,
      Test

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
      indices = (i, j, k, l)
      indset = IndexSet(indices)
      inner = sprint.(show, indices)
      @test sprint(show, indset) == string("IndexSet(", join(inner, ", "), ")")
  end
  @testset "Basic ctors" begin
    I = IndexSet(i,j,k)
    @test IndexSet(I) === I
    @test l ∈ IndexSet(I, l) 
    @test l ∈ IndexSet(l, I)
    @test l ∈ IndexSet( (I, IndexSet(l)) )
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
  end
  @testset "Set operations" begin
    I1 = IndexSet(i,j,k)
    I2 = IndexSet(k,l)
    I3 = IndexSet(j,l)
    @test hassameinds(I1,(k,j,i))
    @test uniqueindex(I1,(I2,I3)) == i
    @test uniqueindex(I1,IndexSet(k, j, i)) == Index()
    @test uniqueinds(I1,I2) == IndexSet(i,j)
    @test setdiff(I1,I2) == IndexSet(i,j)
    @test hassameinds(uniqueinds(I1,I2),(j,i))
    @test commoninds(I1,I2) == IndexSet(k)
    @test commonindex(I1,I2) == k
    @test commonindex(I1,IndexSet(l)) == Index()
    @test commoninds(I1,(j,l)) == IndexSet(j)
    @test commonindex(I1,(j,l)) == j
    @test commoninds(I1,(j,k)) == IndexSet(j,k)
    @test hassameinds(commoninds(I1,(j,k,l)),(j,k))
    @test findinds(I1,"i") == IndexSet(i)
    @test findindex(I1,"j") == j
    @test findindex(I1,"l") == Index()
  end
  @testset "commoninds index ordering" begin
    I = IndexSet(i,k,j)
    J = IndexSet(j,l,i)
    # Test that commoninds respects the ordering
    # of the indices in the first IndexSet
    @test commoninds(I,J) == IndexSet(i,j)
    @test commoninds(J,I) == IndexSet(j,i)
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

    mapprime!(J,1,5)
    @test J == IndexSet(i,j,k^5)
  end
  @testset "compute_strides" begin
    I = IndexSet(i, j)
    @test ITensors.compute_strides(I) == [1, idim]
  end
end
