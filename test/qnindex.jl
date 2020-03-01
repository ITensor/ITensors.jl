using ITensors,
      Test

@testset "QN Index" begin

  @testset "Array of QN Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    @test hasqns(i)
    @test dim(i) == 3
    @test hastags(i,"i")
  end

  @testset "Vararg Constructor" begin
    i = Index(QN(0)=>1,QN(1)=>2;tags="i")
    @test dim(i) == 3
    @test hastags(i,"i")
    @test dir(i) == Out

    j = Index(QN(0)=>1,QN(1)=>2;tags="j",dir=In)
    @test dim(j) == 3
    @test hastags(j,"j")
    @test dir(j) == In
  end

end
