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
    I2 = IndexSet(j,k,l)
    @test hassameinds(I1,(k,j,i))
    @test uniqueindex(I1,I2) == i
    @test uniqueinds(I1,I2) == IndexSet(i)
    @test commoninds(I1,I2) == IndexSet(j,k)
    @test commoninds(I1,(j,l)) == IndexSet(j)
    @test commonindex(I1,(j,l)) == j
    @test findinds(I1,"i") == IndexSet(i)
    @test findindex(I1,"j") == j
  end
end
