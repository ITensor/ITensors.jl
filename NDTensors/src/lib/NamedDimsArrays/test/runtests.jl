using Test: @testset

@testset "NamedDimsArrays" begin
  filenames = filter(readdir(@__DIR__)) do filename
    return startswith("test_")(filename) && endswith(".jl")(filename)
  end
  @testset "Test $(@__DIR__)/$filename" for filename in filenames
    println("Running $(@__DIR__)/$filename")
    @time include(filename)
  end
end
