using ITensors, Test

@testset "Tensors.jl" begin
    @testset "$filename" for filename in (
        "dense.jl",
        "blocksparse.jl",
        "diag.jl"
    )
      println("Running $filename")
      include(filename)
    end
end
