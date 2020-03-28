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
    indset = IndexSet(i,j,k)
    @test length(sprint(show, indset)) > 1
  end
  @testset "Basic constructors" begin
    I = IndexSet(i,j,k)
    @test IndexSet(I) === I
    @test l ∈ IndexSet(I, l) 
    @test l ∈ IndexSet(l, I)
    @test l ∈ IndexSet( (I, IndexSet(l)) )
    #TODO: what should size(::IndexSet) do?
    #@test size(I) == (3,)
    @test length(IndexSet{2}()) == 2
    @test length(IndexSet(Val(2))) == 2
  end
  @testset "length of IndexSet and friends" begin
    @test length(typeof(IndexSet{2}())) == 2
    @test order(IndexSet(Val(2))) == 2
    @test ndims(IndexSet(Val(2))) == 2
    @test ndims(typeof(IndexSet(Val(2)))) == 2
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
    I1 = IndexSet(i,j,k)
    I2 = IndexSet(k,l)
    I3 = IndexSet(j,l)
    @test hassameinds(I1,(k,j,i))
    @test firstsetdiff(I1,I2,I3) == i
    @test isnothing(firstsetdiff(I1,IndexSet(k, j, i)))
    @test setdiff(I1,I2) == [i,j]
    @test hassameinds(setdiff(I1,I2),IndexSet(i,j))
    @test hassameinds(setdiff(I1,I2),(j,i))
    @test I1 ∩ I2 == [k]
    @test hassameinds(I1 ∩ I2,IndexSet(k))
    @test firstintersect(I1,I2) == k
    @test isnothing(firstintersect(I1,IndexSet(l)))
    @test intersect(I1,(j,l)) == [j]
    @test hassameinds(intersect(I1,(j,l)),IndexSet(j))
    @test firstintersect(I1,(j,l)) == j
    @test intersect(I1,(j,k)) == [j,k]
    @test hassameinds(intersect(I1,(j,k)),IndexSet(j,k))
    @test hassameinds(intersect(I1,(j,k,l)),(j,k))
    @test filter(I1,"i") == IndexSet(i)
    @test hassameinds(filter(I1,"i"),IndexSet(i))
    @test getfirst(I1,"j") == j
    @test isnothing(getfirst(I1,"l"))
    @test findfirst(I1,i) == 1
    @test findfirst(I1,j) == 2
    @test findfirst(I1,k) == 3
    @test isnothing(findfirst(I1,Index(2)))
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
    @test strides(I) == (1, idim)
    @test stride(I, 1) == 1
    @test stride(I, 2) == idim
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
end
