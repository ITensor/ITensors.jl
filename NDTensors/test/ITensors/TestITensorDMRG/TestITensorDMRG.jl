## This is getting closer still working on it. No need to review
## Failing for CUDA mostly with eigen (I believe there is some noise in
## eigen decomp with CUBLAS to give slightly different answer than BLAS)
module TestITensorDMRG
using ITensors
using NDTensors

cpu_2_1e3_0_elt = Dict([
  (Float32, Float32(-2)),
  (ComplexF32, Float32(-2)),
  (Float64, Float64(-2)),
  (ComplexF64, Float64(-2)),
])
cpu_2_1e3_1e12_elt = Dict([
  (Float32, Float32(-2)),
  (ComplexF32, Float32(-2)),
  (Float64, Float64(-2)),
  (ComplexF64, Float64(-2)),
])
cpu_2_1e13_0_elt = Dict([
  (Float32, Float32(-2)),
  (ComplexF32, Float32(-2)),
  (Float64, Float64(-2)),
  (ComplexF64, Float64(-2)),
])
cpu_2_1e13_1e12_elt = Dict([
  (Float32, Float32(-2)),
  (ComplexF32, Float32(-2)),
  (Float64, Float64(-2)),
  (ComplexF64, Float64(-2)),
])
cpu_8_1e3_0_elt = Dict([
  (Float32, -10.073674),
  (ComplexF32, -10.075782),
  (Float64, -10.075497754329614),
  (ComplexF64, -10.074342093827271),
])
cpu_8_1e3_1e12_elt = Dict([
  (Float32, -10.0757885),
  (ComplexF32, -10.075473),
  (Float64, -10.073420413855567),
  (ComplexF64, -10.07548425264575),
])
cpu_8_1e13_0_elt = Dict([
  (Float32, -10.124351),
  (ComplexF32, -10.124425),
  (Float64, -10.124596033315136),
  (ComplexF64, -10.124530685719115),
])
cpu_8_1e13_1e12_elt = Dict([
  (Float32, -10.124598),
  (ComplexF32, -10.1246),
  (Float64, -10.124610860111224),
  (ComplexF64, -10.124628100679796),
])

# cuda_2_1e3_0_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_2_1e3_1e12_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_2_1e13_0_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_2_1e13_1e12_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_8_1e3_0_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_8_1e3_1e12_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_8_1e13_0_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])
# cuda_8_1e13_1e12_elt = Dict([(Float32, 1), (ComplexF32, 2), (Float64, 3), (ComplexF64, 4)])

cpu_2_1e3_noise = Dict([(0, cpu_2_1e3_0_elt), (1e-12, cpu_2_1e3_1e12_elt)])
cpu_2_1e13_noise = Dict([(0, cpu_2_1e13_0_elt), (1e-12, cpu_2_1e13_1e12_elt)])
cpu_8_1e3_noise = Dict([(0, cpu_8_1e3_0_elt), (1e-12, cpu_8_1e3_1e12_elt)])
cpu_8_1e13_noise = Dict([(0, cpu_8_1e13_0_elt), (1e-12, cpu_8_1e13_1e12_elt)])

# cuda_2_1e3_noise = Dict([(0, cuda_2_1e3_0_elt), (1e-12, cuda_2_1e3_1e12_elt)])
# cuda_2_1e13_noise = Dict([(0, cuda_2_1e13_0_elt), (1e-12, cuda_2_1e13_1e12_elt)])
# cuda_8_1e3_noise = Dict([(0, cuda_8_1e3_0_elt), (1e-12, cuda_8_1e3_1e12_elt)])
# cuda_8_1e13_noise = Dict([(0, cuda_8_1e13_0_elt), (1e-12, cuda_8_1e13_1e12_elt)])

cpu_2_cut = Dict([(1e-3, cpu_2_1e3_noise), (1e-13, cpu_2_1e13_noise)])
cpu_8_cut = Dict([(1e-3, cpu_8_1e3_noise), (1e-13, cpu_8_1e13_noise)])

# cuda_2_cut = Dict([(1e-3, cuda_2_1e3_noise), (1e-13, cuda_2_1e13_noise)])
# cuda_8_cut = Dict([(1e-3, cuda_2_1e3_noise), (1e-13, cuda_2_1e13_noise)])

cpu_sites = Dict([(2, cpu_2_cut), (8, cpu_8_cut)])
cuda_sites = Dict([(2, cpu_2_cut), (8, cpu_8_cut)])

ref = Dict([(NDTensors.cpu, cpu_sites), (NDTensors.cu, cpu_sites)])

function get_ref_value(device, sites, cutoff, noise, elt)
  return ref[device][sites][cutoff][noise][elt]
end

include("dmrg.jl")
end
