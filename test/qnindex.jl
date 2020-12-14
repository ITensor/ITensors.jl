using ITensors,
      Test

import ITensors: In, Out, Neither

@testset "QN Index" begin

  @testset "hasqns function" begin
    i = Index(4,"i")
    @test hasqns(i) == false
    j = Index(QN(0)=>1,QN(1)=>1)
    @test hasqns(j) == true
  end

  @testset "Array of QN Constructor" begin
    i = Index([QN(0)=>1,QN(1)=>2],"i")
    @test hasqns(i)
    @test dim(i) == 3
    @test hastags(i,"i")
  end

  @testset "Vararg Constructor" begin
    i = Index(QN(0)=>1,QN(1)=>2;tags="i")
    @test hasqns(i)
    @test dim(i) == 3
    @test hastags(i,"i")
    @test dir(i) == Out

    j = Index(QN(0)=>1,QN(1)=>2;tags="j",dir=In)
    @test hasqns(j)
    @test dim(j) == 3
    @test hastags(j,"j")
    @test dir(j) == In
  end

  @testset "flux and qn" begin
    i = dag(Index([QN(0)=>2, QN(1)=>2], "i"))

    @test flux(i => 1) == QN(0)
    @test flux(i => 2) == QN(0)
    @test flux(i => 3) == QN(-1)
    @test flux(i => 4) == QN(-1)
    @test flux(i => Block(1)) == QN(0)
    @test flux(i => Block(2)) == QN(-1)

    @test qn(i => 1) == QN(0)
    @test qn(i => 2) == QN(0)
    @test qn(i => 3) == QN(1)
    @test qn(i => 4) == QN(1)
    @test qn(i => Block(1)) == QN(0)
    @test qn(i => Block(2)) == QN(1)

  end
end

nothing
