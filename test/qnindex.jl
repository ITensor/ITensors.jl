using ITensors, Test

import ITensors: In, Out, Neither

@testset "QN Index" begin
  @testset "hasqns function" begin
    i = Index(4, "i")
    @test hasqns(i) == false
    j = Index(QN(0) => 1, QN(1) => 1)
    @test hasqns(j) == true
  end

  @testset "Array of QN Constructor" begin
    i = Index([QN(0) => 1, QN(1) => 2], "i")
    @test hasqns(i)
    @test dim(i) == 3
    @test hastags(i, "i")
  end

  @testset "Vararg Constructor" begin
    i = Index(QN(0) => 1, QN(1) => 2; tags="i")
    @test hasqns(i)
    @test dim(i) == 3
    @test hastags(i, "i")
    @test dir(i) == Out
    @test dir(i => 2) == Out

    j = Index(QN(0) => 1, QN(1) => 2; tags="j", dir=In)
    @test hasqns(j)
    @test dim(j) == 3
    @test hastags(j, "j")
    @test dir(j) == In
    @test dir(j => 2) == In
  end

  @testset "flux and qn" begin
    i = dag(Index([QN(0) => 2, QN(1) => 2], "i"))

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

  @testset "directsum" begin
    i = Index([QN(0) => 1, QN(1) => 2], "i")
    j = Index([QN(2) => 3, QN(3) => 4], "j")
    ij = ITensors.directsum(i, j; tags="test")
    @test dim(ij) == dim(i) + dim(j)
    @test hastags(ij, "test")
    @test flux(ij, Block(1)) == QN(0)
    @test flux(ij, Block(2)) == QN(1)
    @test flux(ij, Block(3)) == QN(2)
    @test flux(ij, Block(4)) == QN(3)
    @test dim(ij, Block(1)) == 1
    @test dim(ij, Block(2)) == 2
    @test dim(ij, Block(3)) == 3
    @test dim(ij, Block(4)) == 4
    @test_throws ErrorException ITensors.directsum(i, dag(j))
  end
end

nothing
