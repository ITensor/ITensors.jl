using SafeTestsets: @safetestset

@safetestset "NDTensors" begin
  using Test: @testset
  @testset "$(@__DIR__)" begin
    filenames = filter(readdir(@__DIR__)) do f
      startswith("test_")(f) && endswith(".jl")(f)
    end
    for dir in ["lib/", "arraytensor/", "ITensors/"]
      push!(filenames, dir * "runtests.jl")
    end
    @testset "Test $(@__DIR__)/$filename" for filename in filenames
      println("Running $(@__DIR__)/$filename")
      include(filename)
    end
  end
  if "cuda" in ARGS || "all" in ARGS
    using NDTensors
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorCUDA.jl"))
  end
  if "metal" in ARGS || "all" in ARGS
    using NDTensors
    include(joinpath(pkgdir(NDTensors), "ext", "examples", "NDTensorMetal.jl"))
  end
end

nothing
