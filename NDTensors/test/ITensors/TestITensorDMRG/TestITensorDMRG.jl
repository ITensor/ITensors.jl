module TestITensorDMRG
  using ITensors
  using NDTensors

  ref = Dict([(NDTensors.cpu, cpu_cutoff), (NDTensors.cu, cuda_cutoff)])

  cpu_cutoff = Dict([(1e-3, cpu_1e3_noise), (1e-13, cpu_1e13_noise)])
  cuda_cutoff = Dict([(1e-3, cuda_1e3_noise), (1e-13, cuda_1e13_noise)])
  
  cpu_1e3_noise = Dict([(0, cpu_1e3_0_elt), (1e-12, cpu_1e3_1e12_elt)])
  cpu_1e13_noise = Dict([(0, cpu_1e13_0_elt), (1e-12, cpu_1e13_1e12_elt)])
  
  cuda_1e13_noise = Dict([(0, cuda_1e3_0_elt), (1e-12, cuda_1e3_1e12_elt)])
  cuda_1e13_noise = Dict([(0, cuda_1e13_0_elt), (1e-12, cuda_1e13_1e12_elt)])
  
  cpu_1e3_0_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cpu_1e3_1e12_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cpu_1e13_0_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cpu_1e13_1e12_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])

  cuda_1e3_0_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cuda_1e3_1e12_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cuda_1e13_0_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])
  cuda_1e13_1e12_elt = Dict([(Float32, ), (ComplexF32, ), (Float64, ), (ComplexF64, )])

  function get_ref_value(device, cutoff, noise, elt)
    ref[device][cutoff][noise][elt]
  end

  include("dmrg.jl")
end