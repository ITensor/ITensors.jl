using ITensors
using Test

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "$(@__DIR__)" begin
  dirs = ["ext/ITensorMPSChainRulesCoreExt", "Ops", "base"]
  for dir in dirs
    @time include(joinpath(@__DIR__, dir, "runtests.jl"))
  end
end
