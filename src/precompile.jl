function _precompile_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

  ###################################################################################
  # ITensors
  #

  #
  # TagSet
  #

  @assert Base.precompile(Tuple{Type{TagSet},String})   # time: 0.03651878
  @assert Base.precompile(Tuple{Type{SmallString},String})   # time: 0.020327887

  #
  # QN
  #

  @assert Base.precompile(Tuple{Type{QN},Int64})   # time: 0.22320782

  #
  # Index
  #

  @assert Base.precompile(Tuple{Type{Index},Int64})   # time: 0.001648419
  @assert Base.precompile(Tuple{typeof(adjoint),Index{Int64}})   # time: 0.013917884

  #
  # ITensor constructor
  #

  # TODO: generalize
  @assert Base.precompile(Tuple{Type{ITensor},NDTensors.DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 0.001092367
  # TODO: generalize
  @assert Base.precompile(Tuple{Type{ITensor},BlockSparseTensor{Float64, 2, Tuple{Index{Vector{Pair{QN, Int64}}}, Index{Vector{Pair{QN, Int64}}}}, BlockSparse{Float64, Vector{Float64}, 2}}})   # time: 0.026218822
  for IndexT in (Index{Int}, Index{Vector{Pair{QN, Int}}}), N in 1:8
    @assert Base.precompile(Tuple{typeof(randomITensor),Vararg{IndexT,N}})   # time: 0.16231188
  end
  # TODO: generalize
  @assert Base.precompile(Tuple{typeof(tensor),Dense{Float64, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}},Tuple{Int64, Int64}})   # time: 0.001080036



  #
  # ITensor/Tensor priming and tagging
  #

  @assert Base.precompile(Tuple{typeof(adjoint),ITensor})   # time: 0.002728018
  # TODO: generalize
  @assert Base.precompile(Tuple{typeof(prime),DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 0.003458208
  # TODO: generalize
  @assert Base.precompile(Tuple{typeof(prime),BlockSparseTensor{Float64, 2, Tuple{Index{Vector{Pair{QN, Int64}}}, Index{Vector{Pair{QN, Int64}}}}, BlockSparse{Float64, Vector{Float64}, 2}}})   # time: 0.003214042


  #
  # ITensor contraction
  #

  @assert Base.precompile(Tuple{typeof(*),ITensor,ITensor})   # time: 0.43056896

  Ns = 0:6
  ElTypes = (Float64, ComplexF64)
  StorageTypes = (Dense, BlockSparse)
  storagetype(StorageType::Type{Dense}, ElType, N) = StorageType{ElType,Vector{ElType}}
  storagetype(StorageType::Type{BlockSparse}, ElType, N) = StorageType{ElType,Vector{ElType},N}
  indextype(StorageType::Type{Dense}) = Index{Int}
  indextype(StorageType::Type{BlockSparse}) = Index{Vector{Pair{QN,Int64}}}
  for N1 in Ns, N2 in Ns, ElType1 in ElTypes, ElType2 in ElTypes, StorageType in StorageTypes
    IndexType1 = indextype(StorageType)
    IndexType2 = indextype(StorageType)
    StorageType1 = storagetype(StorageType, ElType1, N1)
    StorageType2 = storagetype(StorageType, ElType2, N2)
    IndexSetType1 = NTuple{N1,IndexType1}
    IndexSetType2 = NTuple{N2,IndexType2}
    TensorType1 = Tensor{ElType1, N1, StorageType1, IndexSetType1}
    TensorType2 = Tensor{ElType2, N2, StorageType2, IndexSetType2}
    @assert Base.precompile(Tuple{typeof(ITensors._contract),TensorType1,TensorType2})   # time: 2.0952775
  end

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

  for ElType in (Float64, ComplexF64)
    @assert Base.precompile(Tuple{typeof(randn),Type{Dense{ElType, VecT} where VecT<:(AbstractVector{T} where T)},Int})   # time: 0.018067434
    @assert Base.precompile(Tuple{typeof(randn!),Dense{ElType,Vector{ElType}}})   # time: 0.01703788
    for N in 0:6
      @assert Base.precompile(Tuple{typeof(randn!),BlockSparse{ElType,Vector{ElType},N}})   # time: 0.02958292
    end
  end

  #
  # Dense contraction
  #

  @assert Base.precompile(Tuple{typeof(NDTensors._gemm!),Char,Char,Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,Base.ReshapedArray{Float64, 2, Matrix{Float64}, Tuple{}}})   # time: 0.21759473
  @assert Base.precompile(Tuple{typeof(NDTensors._gemm!),Char,Char,Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}})   # time: 0.12897967
  @assert Base.precompile(Tuple{typeof(NDTensors._contract_scalar_perm!),Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Tuple{Int64, Int64}, Float64, Float64})   # time: 0.004017614
  @assert Base.precompile(Tuple{typeof(NDTensors._contract_scalar_perm!),Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Base.ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}, Tuple{Int64, Int64}, Float64, Int64})   # time: 0.27545762

  #
  # BlockSparse contraction
  #

  @assert Base.precompile(Tuple{typeof(NDTensors.contract_blocks),Block{2},Tuple{Int64, Int64},Block{2},Tuple{Int64, Int64},Val{2}})   # time: 0.002498605

  # block contractions
  Ns = 0:5
  for ElType in (Float64, ComplexF64), N1 in Ns, N2 in Ns, NR in Ns
    ElType1 = ElType
    ElType2 = ElType
    ScalarType1 = Float64
    ScalarType2 = Float64
    ElTypeR = promote_type(ElType1, ElType2, ScalarType1, ScalarType2)
    LabelsType1 = NTuple{N1,Int}
    LabelsType2 = NTuple{N2,Int}
    LabelsTypeR = NTuple{NR,Int}
    IndexTupleType1 = NTuple{N1,Int}
    IndexTupleType2 = NTuple{N2,Int}
    IndexTupleTypeR = NTuple{NR,Int}
    DataType1 = SubArray{ElType1,1,Vector{ElType1},Tuple{UnitRange{Int}},true}
    DataType2 = SubArray{ElType2,1,Vector{ElType2},Tuple{UnitRange{Int}},true}
    DataTypeR = SubArray{ElTypeR,1,Vector{ElTypeR},Tuple{UnitRange{Int}},true}
    StorageType1 = Dense{ElType1,DataType1}
    StorageType2 = Dense{ElType2,DataType2}
    StorageTypeR = Dense{ElTypeR,DataTypeR}
    TensorType1 = Tensor{ElType1,N1,StorageType1,IndexTupleType1}
    TensorType2 = Tensor{ElType2,N2,StorageType2,IndexTupleType2}
    TensorTypeR = Tensor{ElTypeR,NR,StorageTypeR,IndexTupleTypeR}
    @assert Base.precompile(Tuple{typeof(NDTensors.contract!),TensorTypeR,LabelsTypeR,TensorType1,LabelsType1,TensorType2,LabelsType2,ScalarType1,ScalarType2})
  end

end

