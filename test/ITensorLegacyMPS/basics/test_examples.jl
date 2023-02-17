using ITensors, Test, Suppressor

@testset "Example Codes" begin
  examples_dir = joinpath(pkgdir(ITensors), "examples")

  @testset "DMRG with Observer" begin
    @test_nowarn begin
      @capture_out begin
        include(joinpath(examples_dir, "dmrg", "1d_ising_with_observer.jl"))
      end
    end
  end

  @testset "Package Compile Code" begin
    @test_nowarn begin
      @capture_out begin
        include(
          joinpath(pkgdir(ITensors), "src", "packagecompile", "precompile_itensors.jl")
        )
      end
    end
  end
end

nothing
