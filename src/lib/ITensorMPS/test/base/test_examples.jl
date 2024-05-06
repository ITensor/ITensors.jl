@eval module $(gensym())
using ITensors: ITensors
using Suppressor: @capture_out
using Test: @test_nowarn, @testset
@testset "Example Codes" begin
  @testset "DMRG with Observer" begin
    @test_nowarn begin
      @capture_out begin
        include(
          joinpath(
            pkgdir(ITensors),
            "src",
            "lib",
            "ITensorMPS",
            "examples",
            "dmrg",
            "1d_ising_with_observer.jl",
          ),
        )
      end
    end
  end
  @testset "Package Compile Code" begin
    @test_nowarn begin
      @capture_out begin
        include(
          joinpath(
            pkgdir(ITensors), "ext", "ITensorsPackageCompilerExt", "precompile_itensors.jl"
          ),
        )
      end
    end
  end
end
end
