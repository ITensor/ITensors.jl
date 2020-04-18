using ITensors,
      Test,
      Suppressor

@testset "Example Codes" begin

  @testset "Basic Ops" begin
    @test_nowarn begin 
      @capture_out begin
        include("../examples/basic_ops/basic_ops.jl")
      end 
    end
  end

  @testset "DMRG with Observer" begin
    @test_nowarn begin 
      @capture_out begin
        include("../examples/dmrg/dmrg_with_observer.jl")
      end 
    end
  end

end

nothing
