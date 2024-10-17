using SafeTestsets: @safetestset

##TODO there is an issue with Metal and BlockSparse
@safetestset "Downstream tests for ITensor DMRG" begin
  using Test: @testset
  include("TestITensorDMRG/TestITensorDMRG.jl")
  include("../NDTensorsTestUtils/NDTensorsTestUtils.jl")
  using .NDTensorsTestUtils: devices_list, is_supported_eltype
  @testset "Test DMRG $dev, $conserve_qns, $elt, $N" for dev in devices_list(ARGS),
    conserve_qns in [false, true], elt in (Float32, ComplexF32, Float64, ComplexF64),
    N in [4, 10]

    if !is_supported_eltype(dev, elt)
      continue
    end
    if TestITensorDMRG.is_broken(dev, elt, Val(conserve_qns))
      # TODO: Switch to `@test ... broken=true`, introduced
      # in Julia 1.7.
      @test_broken TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns)
    else
      TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns, outputlevel=0)
    end
  end
  using ITensors.ITensorsNamedDimsArraysExt: to_nameddimsarray
  ## Without this line this test was throwing an error of ``NDTensors` not defined`
  using NDTensors: NDTensors
  @testset "Test DMRG with NamedDimsArrays" for dev in (NDTensors.cpu,),
    conserve_qns in [false], elt in (Float32, Float64),
    N in [4, 10]

    dev = dev ∘ to_nameddimsarray
    # TODO: Investigate why this isn't accurate.
    TestITensorDMRG.test_dmrg(elt, N; dev, conserve_qns, rtol_scale=10^3, outputlevel=0)
  end
end
