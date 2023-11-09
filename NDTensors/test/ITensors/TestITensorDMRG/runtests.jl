using Test
using NDTensors
include("TestITensorDMRG.jl")

include("../../device_list.jl")

@testset "Testing DMRG different backends" begin
  for dev in devices_list(ARGS),
    conserve_qns in [false, true],
    elt in (Float32, ComplexF32, Float64, ComplexF64),
    N in [4, 10]

    if !TestITensorDMRG.is_supported_eltype(dev, elt)
      continue
    end
    TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns)
  end
end
