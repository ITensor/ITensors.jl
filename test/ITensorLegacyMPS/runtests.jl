using Test

@testset "$(@__DIR__)" begin
  dirs = [
    "ITensorChainRules",
    "Ops",
    "basics",
  ]
  for dir in dirs
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
