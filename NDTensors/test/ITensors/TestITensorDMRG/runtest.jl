using Test
using NDTensors
include("TestITensorDMRG.jl")

include(joinpath(pkgdir(NDTensors), "test", "device_list.jl"))

@testset "Testing DMRG different backends" begin
  for dev in devices_list(ARGS),
    N in [4, 10],
    elt in (Float32, ComplexF32, Float64, ComplexF64)

    TestITensorDMRG.test_dmrg(elt, N, dev)
  end
end
