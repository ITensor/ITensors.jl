using ITensors, Test, Suppressor

@testset "Example Codes" begin
  examples_dir = joinpath(pkgdir(ITensors), "examples")
  @testset "Basic Ops" begin
    @test_nowarn begin
      @capture_out begin
        include(joinpath(examples_dir, "basic_ops", "basic_ops.jl"))
      end
    end
  end
end

nothing
