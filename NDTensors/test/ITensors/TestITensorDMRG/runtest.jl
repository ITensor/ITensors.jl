using Test
using NDTensors
include("TestITensorDMRG.jl")

include(joinpath(pkgdir(NDTensors), "test", "device_list.jl"))
devs = devices_list(ARGS)

@testset "Testing DMRG different backends" begin
  for dev in devs,
    N in [2, 8],
    cut in [1e-3, 1e-13],
    no in [0, 1e-12],
    elt in [Float32, ComplexF32, Float64, ComplexF64]

    TestITensorDMRG.test_dmrg(elt, N, dev, cut, no)
  end
end
