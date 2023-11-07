using Test
using NDTensors
include("TestITensorDMRG.jl")

include(joinpath(pkgdir(NDTensors), "test", "device_list.jl"))
devs = devices_list(ARGS)

@testset "Testing DMRG different backends" begin
  for dev in devs, N in [4, 8], elt in [Float32, ComplexF32, Float64, ComplexF64]
    TestITensorDMRG.test_dmrg(elt, N, dev)
  end
end
