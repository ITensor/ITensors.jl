@eval module $(gensym())
using ITensors: ITensors
using PackageCompiler: PackageCompiler
using Test: @testset, @test
@testset "ITensorsPackageCompilerExt" begin
  # Testing `ITensors.compile` would take too long so we just check
  # that `ITensorsPackageCompilerExt` overloads `ITensors.compile`.
  @test hasmethod(ITensors.compile, Tuple{ITensors.Algorithm"PackageCompiler"})
end
end
