using Test

@testset "ITensors.ContractionSequenceOptimization" begin
  @testset "$filename" for filename in
                           ["linearalgebra.jl", "dense.jl", "blocksparse.jl", "diag.jl"]
    println("Running $filename")
    include(filename)
  end
end

nothing
