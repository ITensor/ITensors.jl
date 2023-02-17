using Test

@testset "$(@__DIR__)" begin
  filenames = filter(readdir(@__DIR__)) do f
    startswith("test_")(f) && endswith(".jl")(f)
  end
  @testset "Test $(@__DIR__)/$filename" for filename in filenames
    println("Running $(@__DIR__)/$filename")
    @time include(filename)
  end
end
