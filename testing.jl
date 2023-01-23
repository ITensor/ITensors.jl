using Pkg
Pkg.develop("ITensors")
using ITensors, CUDA, Test, Adapt, TimerOutputs, LinearAlgebra, MKL

using NDTensors
BLAS.get_config()
ComplexF64
for SType in (Float64, )
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")

  A = randomCuITensor(SType, i, j, k, l)
  U_gpu, S_gpu, V_gpu = ITensorGPU.svd(A, (j, l))
  U_cpu, S_cpu, V_cpu = svd(cpu(A), (j, l))
  #@show NDTensors.data(storage(cpu(U_gpu))) - NDTensors.data(storage(U_cpu))
  #@show NDTensors.data(storage(cpu(S_gpu))), NDTensors.data(storage(S_cpu))
  @test cpu(U_gpu) * cpu(S_gpu) * cpu(V_gpu) ≈ U_cpu * S_cpu * V_cpu
end

ElT = Float32
VectorType = Vector{ElT}
DenseType = NDTensors.Dense{ElT, VectorType}
@show NDTensors.datatype(DenseType)
similar(VectorType, 20)
zero(eltype)

function zeros_test(datatype::Type{<:AbstractArray{ElT}}, dim::Integer = 0) where {ElT<:Number}
  return fill!(datatype(undef, dim), zero(ElT))
end

function zeros_test(datatype::Type{<:AbstractArray}, dim::Integer = 0) 
  zeros_test(datatype{Float32}, dim)
end
zeros_test(CuVector, 20)
zeros(NDTensors.Dense{ComplexF16, CuVector{ComplexF16}}, 20)

NDTensors.Dense{Float32}(30)
@show undef
NDTensors.Dense{String, Vector{String}}(20)

NDTensors.Dense{Float32}()

v = CuVector{Float32}(undef, 20)
NDTensors.Dense(v)
NDTensors.Dense(Float32, 20)
NDTensors.Dense(ComplexF16)