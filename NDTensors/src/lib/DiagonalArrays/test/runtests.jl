@eval module $(gensym())
using Test: @test, @testset
using NDTensors.DiagonalArrays: DiagonalArrays
@testset "Test NDTensors.DiagonalArrays" begin
  @testset "README" begin
    @test include(
      joinpath(
        pkgdir(DiagonalArrays), "src", "lib", "DiagonalArrays", "examples", "README.jl"
      ),
    ) isa Any
  end
  @testset "Basics" begin
    using NDTensors.DiagonalArrays: diaglength
    a = fill(1.0, 2, 3)
    @test diaglength(a) == 2
    a = fill(1.0)
    @test diaglength(a) == 1
  end
end
end
