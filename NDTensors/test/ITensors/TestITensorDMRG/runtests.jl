using Test
using NDTensors
## TODO headergaurd
include("../../NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: NDTensorsTestUtils
include("TestITensorDMRG.jl")

@testset "Test DMRG $dev, $conserve_qns, $elt, $N" for dev in NDTensorsTestUtils.devices_list(ARGS),
  conserve_qns in [false, true],
  elt in (Float32, ComplexF32, Float64, ComplexF64),
  N in [4, 10]

  if !NDTensorsTestUtils.is_supported_eltype(dev, elt)
    continue
  end
  if TestITensorDMRG.is_broken(dev, elt, Val(conserve_qns))
    # TODO: Switch to `@test ... broken=true`, introduced
    # in Julia 1.7.
    @test_broken TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns)
  else
    TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns)
  end
end
