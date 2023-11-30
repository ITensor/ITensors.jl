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
end
end
