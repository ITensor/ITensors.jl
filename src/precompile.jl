function _precompile_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

  ###################################################################################
  # ITensors
  #

  #
  # TagSet
  #

  @assert Base.precompile(Tuple{Type{TagSet},String})   # time: 0.03651878

  #
  # Index
  #

  @assert Base.precompile(Tuple{Type{Index},Int64})   # time: 0.001648419
  @assert Base.precompile(Tuple{typeof(adjoint),Index{Int64}})   # time: 0.013917884

  #
  # ITensor
  #

  @assert Base.precompile(Tuple{Type{ITensor},NDTensors.DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 0.001092367
  @assert Base.precompile(Tuple{typeof(adjoint),ITensor})   # time: 0.002728018
  @assert Base.precompile(Tuple{typeof(randomITensor),Index{Int64},Index{Int64}})   # time: 0.16231188
  @assert Base.precompile(Tuple{typeof(prime),DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 0.003458208

  #
  # ITensor contraction
  #

  @assert Base.precompile(Tuple{typeof(*),ITensor,ITensor})   # time: 0.43056896
  @assert Base.precompile(Tuple{typeof(_contract),DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}},DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 2.0952775

  ###################################################################################
  # ITensors.NDTensors
  #

  #
  # Tuple tools
  #

  Base.precompile(Tuple{typeof(permute),Tuple{Int64, Int64},Tuple{Union{Nothing, Int64}, Union{Nothing, Int64}}})   # time: 0.001209587

  #
  # Contraction properties
  #

  @assert Base.precompile(Tuple{typeof(NDTensors.compute_perms!),NDTensors.ContractionProperties{2, 2, 2}})   # time: 0.02025035
  @assert Base.precompile(Tuple{Type{NDTensors.ContractionProperties},Tuple{Int64, Int64},Tuple{Int64, Int64},Tuple{Int64, Int64}})   # time: 0.001703243
  @assert Base.precompile(Tuple{typeof(NDTensors.contract_labels),Type{Val{2}},Tuple{Int64, Int64},Tuple{Int64, Int64}})   # time: 0.037068952



  #
  # Storage types
  #

  @assert Base.precompile(Tuple{typeof(randn),Type{Dense{Float64, VecT} where VecT<:(AbstractVector{T} where T)},Int64})   # time: 0.018067434
  @assert Base.precompile(Tuple{typeof(randn!),Dense{Float64, Vector{Float64}}})   # time: 0.01703788

  #
  # Dense contraction
  #

  @assert Base.precompile(Tuple{typeof(NDTensors._gemm!),Char,Char,Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,Base.ReshapedArray{Float64, 2, Matrix{Float64}, Tuple{}}})   # time: 0.21759473
  @assert Base.precompile(Tuple{typeof(NDTensors._gemm!),Char,Char,Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}})   # time: 0.12897967
  @assert Base.precompile(Tuple{typeof(NDTensors._contract_scalar_perm!),Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Tuple{Int64, Int64}, Float64, Float64})   # time: 0.004017614
  @assert Base.precompile(Tuple{typeof(NDTensors._contract_scalar_perm!),Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Tuple{Int64, Int64}, Float64, Int64})   # time: 0.27545762

end

