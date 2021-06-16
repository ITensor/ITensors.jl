using Test

@testset "ITensors.ContractionSequenceOptimization" begin
  @testset "$filename" for filename in ["itensor_contract.jl"]
    println("Running $filename")
    include(filename)
  end
end

nothing
