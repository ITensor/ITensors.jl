using ITensors,
      Test,
      Suppressor

@testset "Example Codes" begin

  examples_dir = joinpath("..", "examples")

  @testset "Basic Ops" begin
    @test_nowarn begin 
      @capture_out begin
        include(joinpath(examples_dir, "basic_ops", "basic_ops.jl"))
      end 
    end
  end

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
        include(joinpath("..", "src", "packagecompile", "precompile_itensors.jl"))
      end 
    end
  end

end

nothing
