# COV_EXCL_START
const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
  function getsym(arg)
    isa(arg, Symbol) && return arg
    @assert isa(arg, GlobalRef)
    return arg.name
  end

  f = get(__bodyfunction__, mnokw, nothing)
  if f === nothing
    fmod = mnokw.module
    # The lowered code for `mnokw` should look like
    #   %1 = mkw(kwvalues..., #self#, args...)
    #        return %1
    # where `mkw` is the name of the "active" keyword body-function.
    ast = Base.uncompressed_ast(mnokw)
    if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
      callexpr = ast.code[end - 1]
      if isa(callexpr, Expr) && callexpr.head == :call
        fsym = callexpr.args[1]
        if isa(fsym, Symbol)
          f = getfield(fmod, fsym)
        elseif isa(fsym, GlobalRef)
          if fsym.mod === Core && fsym.name === :_apply
            f = getfield(mnokw.module, getsym(callexpr.args[2]))
          elseif fsym.mod === Core && fsym.name === :_apply_iterate
            f = getfield(mnokw.module, getsym(callexpr.args[3]))
          else
            f = getfield(fsym.mod, fsym.name)
          end
        else
          f = missing
        end
      else
        f = missing
      end
    else
      f = missing
    end
    __bodyfunction__[mnokw] = f
  end
  return f
end

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
  # TODO: generalize
  @assert Base.precompile(Tuple{typeof(_addtag_ordered!),MVector{4,UInt128},Int64,UInt128})   # time: 0.006628992
  @assert Base.precompile(Tuple{typeof(isint),SmallString})   # time: 0.001743865

  #
  # QN
  #

  @assert Base.precompile(Tuple{Type{QN},Int64})   # time: 0.22320782

  #
  # Index
  #

  @assert Base.precompile(Tuple{Type{Index},Int64})   # time: 0.001648419
  @assert Base.precompile(Tuple{typeof(adjoint),Index{Int64}})   # time: 0.013917884
  @assert Base.precompile(Tuple{typeof(dag),Index{Int64}})   # time: 0.001862452
  # TODO: generalize
  @assert Base.precompile(
    Tuple{typeof(_setdiff),Tuple{Index{Int64},Index{Int64}},Vector{Index{Int64}}}
  )   # time: 0.008226005
  @assert Base.precompile(
    Tuple{typeof(_intersect),Tuple{Index{Int64},Index{Int64}},Vector{Index{Int64}}}
  )   # time: 0.003517178

  #
  # ITensor constructor
  #

  # TODO: generalize
  @assert Base.precompile(
    Tuple{
      Type{ITensor},
      NDTensors.DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
    },
  )   # time: 0.001092367
  # TODO: generalize
  @assert Base.precompile(
    Tuple{
      Type{ITensor},
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.026218822
  for IndexT in (Index{Int}, Index{Vector{Pair{QN,Int}}}), N in 1:8
    @assert Base.precompile(Tuple{typeof(randomITensor),Vararg{IndexT,N}})   # time: 0.16231188
  end
  # TODO: generalize
  @assert Base.precompile(
    Tuple{
      typeof(tensor),
      Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      Tuple{Int64,Int64},
    },
  )   # time: 0.001080036
  # TODO: generalize
  @assert Base.precompile(Tuple{typeof(combiner),Index{Int64}})   # time: 0.010823254

  #
  # ITensor/Tensor priming and tagging
  #

  @assert Base.precompile(Tuple{typeof(adjoint),ITensor})   # time: 0.002728018
  # TODO: generalize
  @assert Base.precompile(
    Tuple{
      typeof(prime),
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
    },
  )   # time: 0.003458208
  # TODO: generalize
  @assert Base.precompile(
    Tuple{
      typeof(prime),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.003214042
  @assert Base.precompile(
    Tuple{
      typeof(settags),
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      TagSet,
      Index{Int64},
    },
  )   # time: 0.004762882
  @assert Base.precompile(Tuple{typeof(dag),AllowAlias,ITensor})   # time: 0.001427794
  # TODO: generalize or @nospecialize
  @assert Base.precompile(
    Tuple{typeof(dag),AllowAlias,Tensor{Number,2,Combiner,Tuple{Index{Int64},Index{Int64}}}}
  )   # time: 0.001614791

  #
  # ITensor contraction
  #

  @assert Base.precompile(Tuple{typeof(*),ITensor,ITensor})   # time: 0.43056896

  @assert Base.precompile(Tuple{typeof(contract),ITensor,ITensor,ITensor})   # time: 0.05387871
  @assert Base.precompile(Tuple{typeof(contract),ITensor,ITensor,ITensor,ITensor})   # time: 0.05387871
  @assert Base.precompile(Tuple{typeof(contract),ITensor,ITensor,ITensor,ITensor,ITensor})   # time: 0.05387871

  Ns = 0:6
  ElTypes = (Float64, ComplexF64)
  StorageTypes = (Dense, BlockSparse)
  storagetype(StorageType::Type{Dense}, ElType, N) = StorageType{ElType,Vector{ElType}}
  function storagetype(StorageType::Type{BlockSparse}, ElType, N)
    return StorageType{ElType,Vector{ElType},N}
  end
  indextype(StorageType::Type{Dense}) = Index{Int}
  indextype(StorageType::Type{BlockSparse}) = Index{Vector{Pair{QN,Int64}}}
  for N1 in Ns,
    N2 in Ns,
    ElType1 in ElTypes,
    ElType2 in ElTypes,
    StorageType in StorageTypes

    IndexType1 = indextype(StorageType)
    IndexType2 = indextype(StorageType)
    StorageType1 = storagetype(StorageType, ElType1, N1)
    StorageType2 = storagetype(StorageType, ElType2, N2)
    IndexSetType1 = NTuple{N1,IndexType1}
    IndexSetType2 = NTuple{N2,IndexType2}
    TensorType1 = Tensor{ElType1,N1,StorageType1,IndexSetType1}
    TensorType2 = Tensor{ElType2,N2,StorageType2,IndexSetType2}
    @assert Base.precompile(Tuple{typeof(ITensors._contract),TensorType1,TensorType2})   # time: 2.0952775
  end

  # Contraction with combiner
  for N1 in Ns, N2 in Ns, ElType1 in ElTypes, StorageType in StorageTypes
    ElType2 = Number
    IndexType1 = indextype(StorageType)
    IndexType2 = indextype(StorageType)
    StorageType1 = storagetype(StorageType, ElType1, N1)
    StorageType2 = storagetype(StorageType, ElType2, N2)
    IndexSetType1 = NTuple{N1,IndexType1}
    IndexSetType2 = NTuple{N2,IndexType2}
    TensorType1 = Tensor{ElType1,N1,StorageType1,IndexSetType1}
    TensorType2 = Tensor{ElType2,N2,Combiner,IndexSetType2}
    @assert Base.precompile(Tuple{typeof(ITensors._contract),TensorType1,TensorType2})   # time: 2.0952775
    @assert Base.precompile(Tuple{typeof(ITensors._contract),TensorType2,TensorType1})   # time: 2.0952775
  end

  # TODO: also cover this case, shows up in SVD
  #@assert Base.precompile(Tuple{typeof(_contract),DenseTensor{ComplexF64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{ComplexF64, Base.ReshapedArray{ComplexF64, 1, Adjoint{ComplexF64, Matrix{ComplexF64}}, Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}}}}},Tensor{Number, 2, Combiner, Tuple{Index{Int64}, Index{Int64}}}})   # time: 0.33228973

  #
  # ITensor svd
  #

  # TODO: generalize or @nospecialize
  @assert Base.precompile(Tuple{typeof(svd),ITensor,Index{Int64}})   # time: 0.67957884
  @assert Base.precompile(Tuple{typeof(svd),ITensor,Tuple{Index{Int64},Index{Int64}}})   # time: 0.86975384

  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      DenseTensor{
        Float64,
        2,
        Tuple{Index{Int64},Index{Int64}},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            Adjoint{Float64,Matrix{Float64}},
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
      Tensor{Number,2,Combiner,Tuple{Index{Int64},Index{Int64}}},
    },
  )   # time: 0.5841334
  @assert Base.precompile(Tuple{typeof(NDTensors.svd_recursive),Matrix{Float64}})   # time: 0.55733496

  ###################################################################################
  # ITensors.NDTensors
  #

  #
  # Tuple tools
  #

  Base.precompile(
    Tuple{
      typeof(permute),Tuple{Int64,Int64},Tuple{Union{Nothing,Int64},Union{Nothing,Int64}}
    },
  )   # time: 0.001209587

  #
  # Contraction properties
  #

  @assert Base.precompile(
    Tuple{typeof(NDTensors.compute_perms!),NDTensors.ContractionProperties{2,2,2}}
  )   # time: 0.02025035
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64},
    },
  )   # time: 0.001703243
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),Type{Val{2}},Tuple{Int64,Int64},Tuple{Int64,Int64}
    },
  )   # time: 0.037068952

  #
  # Storage types
  #

  for ElType in (Float64, ComplexF64)
    @assert Base.precompile(
      Tuple{
        typeof(randn),Type{Dense{ElType,VecT} where VecT<:(AbstractVector{T} where {T})},Int
      },
    )   # time: 0.018067434
    @assert Base.precompile(Tuple{typeof(randn!),Dense{ElType,Vector{ElType}}})   # time: 0.01703788
    for N in 0:6
      @assert Base.precompile(Tuple{typeof(randn!),BlockSparse{ElType,Vector{ElType},N}})   # time: 0.02958292
    end
  end

  #
  # Dense contraction
  #

  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Matrix{Float64},Tuple{}},
    },
  )   # time: 0.21759473
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
    },
  )   # time: 0.12897967
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_perm!),
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.004017614
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_perm!),
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Tuple{Int64,Int64},
      Float64,
      Int64,
    },
  )   # time: 0.27545762

  #
  # BlockSparse contraction
  #

  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{2},
      Tuple{Int64,Int64},
      Block{2},
      Tuple{Int64,Int64},
      Val{2},
    },
  )   # time: 0.002498605

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
    @assert Base.precompile(
      Tuple{
        typeof(NDTensors.contract!),
        TensorTypeR,
        LabelsTypeR,
        TensorType1,
        LabelsType1,
        TensorType2,
        LabelsType2,
        ScalarType1,
        ScalarType2,
      },
    )
  end

  ###################################################################################
  # ITensors dmrg
  #

  # TODO: add @assert

  @assert Base.precompile(Tuple{typeof(dmrg),MPO,MPS,Sweeps})   # time: 8.852374
  @assert Base.precompile(Tuple{Type{MPO},AutoMPO,Vector{Index{Int64}}})   # time: 0.19587289
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(replacebond!)),
      NamedTuple{
        (
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :which_decomp,
          :svd_alg,
        ),
        Tuple{Int64,Int64,Float64,Nothing,String,Bool,Nothing,String},
      },
      typeof(replacebond!),
      MPS,
      Int64,
      ITensor,
    },
  )   # time: 0.16070782
  @assert Base.precompile(Tuple{typeof(randomMPS),Vector{Index{Int64}},Int64})   # time: 0.037780974

  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(eigen)),
      NamedTuple{
        (
          :ishermitian,
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Bool,Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(eigen),
      ITensor,
      Vector{Index{Int64}},
      Vector{Index{Int64}},
    },
  )   # time: 0.02893102

  @assert Base.precompile(
    Tuple{
      typeof(_permute),
      NeverAlias,
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      Vector{Index{Int64}},
    },
  )   # time: 0.2220718
  @assert Base.precompile(
    Tuple{
      typeof(_permute),
      NeverAlias,
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.12732054
  @assert Base.precompile(
    Tuple{
      typeof(_permute),
      NeverAlias,
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
      Vector{Index{Int64}},
    },
  )   # time: 0.060088485

  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
    },
  )   # time: 0.18723752

  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      Vector{Index{Int64}},
      Vector{Index{Int64}},
    },
  )   # time: 0.07100896

  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(combiner)),
      NamedTuple{(:dir, :tags),Tuple{Arrow,String}},
      typeof(combiner),
      Index{Int64},
    },
  )   # time: 0.035185322

  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(factorize)),
      NamedTuple{
        (
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(factorize),
      ITensor,
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.035102397

  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Vararg{Tuple{Index{Int64},Index{Int64},Index{Int64}},N} where N,
    },
  )   # time: 0.03414814
  @assert Base.precompile(Tuple{typeof(hash),Index{Int64},UInt64})   # time: 0.02896381
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      EmptyTensor{
        EmptyNumber,
        2,
        Tuple{Index{Int64},Index{Int64}},
        EmptyStorage{EmptyNumber,Dense{EmptyNumber,Vector{EmptyNumber}}},
      },
      Float64,
      Pair{Index{Int64},Int64},
      Pair{Index{Int64},Int64},
    },
  )   # time: 0.028590519
  @assert Base.precompile(Tuple{typeof(+),AutoMPO,Tuple{String,Int64,String,Int64}})   # time: 0.026002342
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
      Vector{Index{Int64}},
      Vector{Index{Int64}},
    },
  )   # time: 0.023155361
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(factorize)),
      NamedTuple{
        (
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(factorize),
      ITensor,
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.02169771
  @assert Base.precompile(Tuple{Type{SmallString},String})   # time: 0.021690434
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      DiagTensor{Float64,2,Tuple{Index{Int64},Index{Int64}},Diag{Float64,Vector{Float64}}},
      Tuple{Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.021272324
  @assert Base.precompile(Tuple{typeof(axpy!),Float64,ITensor,ITensor})   # time: 0.021154549
  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
    },
  )   # time: 0.020586168
  @assert Base.precompile(
    Tuple{
      typeof(replaceind),
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      Index{Int64},
      Index{Int64},
    },
  )   # time: 0.017293353
  @assert Base.precompile(
    Tuple{
      typeof(replaceind),
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      Index{Int64},
      Index{Int64},
    },
  )   # time: 0.017127672
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.016837109
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
    },
  )   # time: 0.014842648
  @assert Base.precompile(Tuple{typeof(permute),ITensor,Vector{Index{Int64}}})   # time: 0.01225709
  @assert Base.precompile(Tuple{typeof(rmul!),ITensor,Float64})   # time: 0.012119683
  @assert Base.precompile(Tuple{typeof(rmul!),ITensor,Bool})   # time: 0.011333919
  @assert Base.precompile(Tuple{typeof(rmul!),ITensor,Int64})   # time: 0.010320051
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),Tuple{Index{Int64},Index{Int64},Index{Int64}},NTuple{4,Index{Int64}}
    },
  )   # time: 0.010004855
  @assert Base.precompile(Tuple{typeof(permute),ITensor,Index{Int64},Index{Int64}})   # time: 0.009674822
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      NTuple{4,Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Vararg{Any,N} where N,
    },
  )   # time: 0.009331419
  @assert Base.precompile(Tuple{typeof(sim),Vector{Index{Int64}}})   # time: 0.007713519
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.007117971
  @assert Base.precompile(Tuple{typeof(setcutoff!),Sweeps,Float64})   # time: 0.00675207
  @assert Base.precompile(Tuple{typeof(setmaxdim!),Sweeps,Int64,Int64,Int64,Int64,Int64})   # time: 0.006474703
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(combiner)),
      NamedTuple{(:dir, :tags),Tuple{Arrow,String}},
      typeof(combiner),
      Index{Int64},
      Index{Int64},
    },
  )   # time: 0.006264304
  @assert Base.precompile(
    Tuple{typeof(==),Tuple{Index{Int64},Index{Int64}},Vector{Index{Int64}}}
  )   # time: 0.006087399
  @assert Base.precompile(Tuple{typeof(permute),Vector{Index{Int64}},Vector{Index{Int64}}})   # time: 0.006076256
  @assert Base.precompile(
    Tuple{typeof(hascommoninds),Vector{Index{Int64}},Vector{Index{Int64}}}
  )   # time: 0.006037176
  @assert Base.precompile(Tuple{typeof(map!),Function,ITensor,ITensor,ITensor})   # time: 0.005631221
  @assert Base.precompile(
    Tuple{typeof(replaceinds),ITensor,Vector{Index{Int64}},Vector{Index{Int64}}}
  )   # time: 0.005545829
  @assert Base.precompile(
    Tuple{
      typeof(noprime),
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
    },
  )   # time: 0.004949224
  @assert Base.precompile(Tuple{typeof(replaceind!),ITensor,Index{Int64},Index{Int64}})   # time: 0.004649281
  @assert Base.precompile(
    Tuple{
      typeof(noprime),
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
    },
  )   # time: 0.004642254
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds!),
      ITensor,
      Tuple{Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.004523673
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      ITensor,
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.004306959
  @assert Base.precompile(
    Tuple{
      typeof(emptyITensor),
      Type{Float64},
      Index{Int64},
      Index{Int64},
      Vararg{Index{Int64},N} where N,
    },
  )   # time: 0.004225701
  @assert Base.precompile(
    Tuple{
      Type{Vector{IndexT} where IndexT<:Index},Index{Int64},Vararg{Index{Int64},N} where N
    },
  )   # time: 0.004044858
  @assert Base.precompile(
    Tuple{
      typeof(prime),
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
    },
  )   # time: 0.003961048
  @assert Base.precompile(Tuple{typeof(siteinds),String,Int64})   # time: 0.003688941
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
      Vararg{Any,N} where N,
    },
  )   # time: 0.003352725
  @assert Base.precompile(Tuple{typeof(_addtag_ordered!),MVector{4,UInt128},Int64,UInt128})   # time: 0.002820313
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      NTuple{4,Index{Int64}},
    },
  )   # time: 0.002760013
  @assert Base.precompile(
    Tuple{
      Type{ITensor},AllowAlias,Matrix{Float64},Index{Int64},Vararg{Index{Int64},N} where N
    },
  )   # time: 0.002670136
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),NTuple{4,Index{Int64}},NTuple{4,Index{Int64}},Vararg{Any,N} where N
    },
  )   # time: 0.002634403
  @assert Base.precompile(Tuple{typeof(emptyITensor),Type{Float64},NTuple{4,Index{Int64}}})   # time: 0.002591185
  @assert Base.precompile(
    Tuple{typeof(emptyITensor),Type{Float64},Tuple{Index{Int64},Index{Int64},Index{Int64}}}
  )   # time: 0.002547633
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      NTuple{4,Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.00254151
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      NTuple{4,Index{Int64}},
      NTuple{4,Index{Int64}},
      Vararg{NTuple{4,Index{Int64}},N} where N,
    },
  )   # time: 0.002390298
  @assert Base.precompile(Tuple{Type{Sweeps},Int64})   # time: 0.002360926
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),Tuple{Index{Int64},Index{Int64}},Tuple{Index{Int64},Index{Int64}}
    },
  )   # time: 0.002323905
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
    },
  )   # time: 0.002021084
  @assert Base.precompile(Tuple{typeof(isint),SmallString})   # time: 0.002003834
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      NonQN,
      EmptyTensor{
        EmptyNumber,
        1,
        Tuple{Index{Int64}},
        EmptyStorage{EmptyNumber,Dense{EmptyNumber,Vector{EmptyNumber}}},
      },
      Float64,
      Int64,
    },
  )   # time: 0.001985102
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      DenseTensor{Float64,4,NTuple{4,Index{Int64}},Dense{Float64,Vector{Float64}}},
    },
  )   # time: 0.001927719
  @assert Base.precompile(Tuple{typeof(toMatrix),Vector{MatElem{Float64}}})   # time: 0.001887898
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      Float64,
      Pair{Index{Int64},Int64},
      Pair{Index{Int64},Int64},
    },
  )   # time: 0.001683615
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.001674245
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Int64}},
      NTuple{4,Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001493712
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      NTuple{4,Index{Int64}},
    },
  )   # time: 0.001435034
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001432821
  @assert Base.precompile(
    Tuple{
      Type{ITensor},AllowAlias,Array{Float64,3},Index{Int64},Vararg{Index{Int64},N} where N
    },
  )   # time: 0.001420816
  @assert Base.precompile(
    Tuple{
      Type{ITensor},
      DenseTensor{
        Float64,
        3,
        Tuple{Index{Int64},Index{Int64},Index{Int64}},
        Dense{Float64,Vector{Float64}},
      },
    },
  )   # time: 0.001393534
  @assert Base.precompile(
    Tuple{typeof(_intersect),NTuple{4,Index{Int64}},NTuple{4,Index{Int64}}}
  )   # time: 0.001372008
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001362485
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64}},
    },
  )   # time: 0.001345698
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),Vector{Index{Int64}},NTuple{4,Index{Int64}},NTuple{4,Index{Int64}}
    },
  )   # time: 0.001248066
  @assert Base.precompile(Tuple{typeof(dag),AllowAlias,ITensor})   # time: 0.001239248
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001227811
  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
      DenseTensor{
        Float64,2,Tuple{Index{Int64},Index{Int64}},Dense{Float64,Vector{Float64}}
      },
    },
  )   # time: 0.001144245
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001132108
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(measure!)),
      NamedTuple{
        (:energy, :psi, :bond, :sweep, :half_sweep, :spec, :outputlevel, :sweep_is_done),
        Tuple{Float64,MPS,Int64,Int64,Int64,Spectrum{Vector{Float64}},Int64,Bool},
      },
      typeof(measure!),
      NoObserver,
    },
  )   # time: 0.001084504
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(checkdone!)),
      NamedTuple{(:energy, :psi, :sweep, :outputlevel),Tuple{Float64,MPS,Int64,Int64}},
      typeof(checkdone!),
      NoObserver,
    },
  )   # time: 0.001022789
  @assert Base.precompile(
    Tuple{
      Type{ITensor},
      AllowAlias,
      Type{Float64},
      Array{Float64,3},
      Tuple{Index{Int64},Index{Int64},Index{Int64}},
    },
  )   # time: 0.001020571

  # XXX: syntax error
  #Base.precompile(Tuple{typeof(op!),ITensor,OpName{Symbol("S-")},ITensors.SiteType{S=1},Index{Int64}})   # time: 0.004136715
  #Base.precompile(Tuple{typeof(op!),ITensor,OpName{:Id},ITensors.SiteType{Generic},Index{Int64}})   # time: 0.020093283
  #Base.precompile(Tuple{typeof(op!),ITensor,OpName{Symbol("S+")},ITensors.SiteType{S=1},Index{Int64}})   # time: 0.005026562
  #Base.precompile(Tuple{typeof(op!),ITensor,OpName{:Sz},ITensors.SiteType{S=1},Index{Int64}})   # time: 0.032288317

  # XXX: precompile failure
  #@assert Base.precompile(Tuple{typeof(copyto!),ITensor,Broadcasted{ITensorOpScalarStyle, _A, typeof(/), Tuple{ITensor, Float64}} where _A})   # time: 0.006613353

  ###################################################################################
  # NDTensors dmrg
  #

  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Array{Float64,3},Tuple{}},
    },
  )   # time: 0.18408749
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Array{Float64,5},Tuple{}},
    },
  )   # time: 0.14643145
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Array{Float64,4},Tuple{}},
    },
  )   # time: 0.13572618
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{Float64,2,Array{Float64,0},Tuple{}},
    },
  )   # time: 0.13372602
  @assert Base.precompile(Tuple{typeof(*),Float64,Dense{Float64,Vector{Float64}}})   # time: 0.040729687
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{3}},
      Tuple{Int64,Int64,Int64},
      NTuple{4,Int64},
    },
  )   # time: 0.03579207
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{5}},
      NTuple{4,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.026385212
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{3}},NTuple{4,Int64},Tuple{Int64}}
  )   # time: 0.023019746
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{0}},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.017579352
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(truncate!)),
      NamedTuple{
        (:maxdim, :cutoff, :use_absolute_cutoff, :use_relative_cutoff),
        Tuple{Int64,Float64,Bool,Bool},
      },
      typeof(truncate!),
      Vector{Float64},
    },
  )   # time: 0.01318219
  @assert Base.precompile(Tuple{typeof(norm),Dense{Float64,Vector{Float64}}})   # time: 0.012643798
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),Type{Val{4}},Tuple{Int64,Int64},Tuple{Int64,Int64}
    },
  )   # time: 0.009895875
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),Type{Val{2}},Tuple{Int64,Int64,Int64},Tuple{Int64}
    },
  )   # time: 0.007162047
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      NTuple{4,Int64},
      Tuple{Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.003799894
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{3}},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64},
    },
  )   # time: 0.003739069
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},NTuple{5,Int64},NTuple{4,Int64},NTuple{5,Int64}
    },
  )   # time: 0.003445886
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{3}},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.003094588
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{3}},
      NTuple{4,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.00309218
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{5}},NTuple{5,Int64},NTuple{4,Int64}}
  )   # time: 0.003011967
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{4}},
      NTuple{5,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.002952535
  @assert Base.precompile(
    Tuple{typeof(NDTensors.intersect_positions),Tuple{Int64,Int64,Int64},Tuple{Int64,Int64}}
  )   # time: 0.002808216
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{4}},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.002597248
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
      NTuple{4,Int64},
    },
  )   # time: 0.002550134
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      NTuple{4,Int64},
      Tuple{Int64,Int64,Int64},
      NTuple{5,Int64},
    },
  )   # time: 0.002516212
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},NTuple{4,Int64},NTuple{4,Int64},NTuple{4,Int64}
    },
  )   # time: 0.002458303
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_labels),
      Type{Val{2}},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.002254591
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{4}},NTuple{4,Int64},NTuple{4,Int64}}
  )   # time: 0.002049893
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{0}},NTuple{4,Int64},NTuple{4,Int64}}
  )   # time: 0.00202476
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.intersect_positions),
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.001726414
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64},
      Tuple{Int64,Int64},
    },
  )   # time: 0.001404129
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.001299593
  @assert Base.precompile(
    Tuple{
      Type{NDTensors.ContractionProperties},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.001215327
  @assert Base.precompile(
    Tuple{Type{NDTensors.ContractionProperties},NTuple{4,Int64},NTuple{4,Int64},Tuple{}}
  )   # time: 0.001178936

  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.drop_singletons),
      NDTensors.Order{2},
      Tuple{Int64,Int64},
      Tuple{Int64,Int64},
    },
  )   # time: 0.002948757
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.drop_singletons),
      NDTensors.Order{2},
      Tuple{Int64,Int64,Int64},
      Tuple{Int64,Int64,Int64},
    },
  )   # time: 0.001650279

  ###################################################################################
  # ITensors dmrg qn
  #

  @assert Base.precompile(Tuple{typeof(dmrg),MPO,MPS,Sweeps})   # time: 5.2061253
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      Tensor{
        Number,
        3,
        Combiner,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
      },
    },
  )   # time: 0.8235568
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Tensor{
        Number,
        3,
        Combiner,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
      },
    },
  )   # time: 0.56411266
  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.3756296
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      Tensor{
        Number,
        3,
        Combiner,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
      },
    },
  )   # time: 0.3239403
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      EmptyTensor{
        EmptyNumber,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        EmptyStorage{EmptyNumber,BlockSparse{EmptyNumber,Vector{EmptyNumber},2}},
      },
      Float64,
      Pair{Index{Vector{Pair{QN,Int64}}},Int64},
      Pair{Index{Vector{Pair{QN,Int64}}},Int64},
    },
  )   # time: 0.14033888
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(replacebond!)),
      NamedTuple{
        (
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :which_decomp,
          :svd_alg,
        ),
        Tuple{Int64,Int64,Float64,Nothing,String,Bool,Nothing,String},
      },
      typeof(replacebond!),
      MPS,
      Int64,
      ITensor,
    },
  )   # time: 0.13230652
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Tensor{
        Number,2,Combiner,Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}}
      },
    },
  )   # time: 0.12175041
  @assert Base.precompile(Tuple{Type{MPO},AutoMPO,Vector{Index{Vector{Pair{QN,Int64}}}}})   # time: 0.11935654
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      Tensor{
        Number,2,Combiner,Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}}
      },
    },
  )   # time: 0.09140348
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      EmptyTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        EmptyStorage{Float64,BlockSparse{Float64,Vector{Float64},3}},
      },
      Float64,
      IndexVal{Index{Vector{Pair{QN,Int64}}}},
      IndexVal{Index{Vector{Pair{QN,Int64}}}},
      IndexVal{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.08036581
  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.07833458
  @assert Base.precompile(
    Tuple{
      typeof(_map!!),
      Function,
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.07004794
  @assert Base.precompile(
    Tuple{
      typeof(replaceind),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Index{Vector{Pair{QN,Int64}}},
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.068822585
  @assert Base.precompile(
    Tuple{typeof(randomMPS),Vector{Index{Vector{Pair{QN,Int64}}}},Vector{String},Int64}
  )   # time: 0.05550305
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(factorize)),
      NamedTuple{
        (
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(factorize),
      ITensor,
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.05197685
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      DiagBlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        DiagBlockSparse{Float64,Vector{Float64},2},
      },
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.05158091
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      DiagBlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        DiagBlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.05121782
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(factorize)),
      NamedTuple{
        (
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(factorize),
      ITensor,
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.029268129
  @assert Base.precompile(Tuple{typeof(_addtag_ordered!),MVector{4,UInt128},Int64,UInt128})   # time: 0.029129894
  @assert Base.precompile(
    Tuple{
      Type{ITensor},
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.025942683
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.021642772
  @assert Base.precompile(Tuple{Type{SmallString},String})   # time: 0.019998804
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      HasQNs,
      EmptyTensor{
        EmptyNumber,
        1,
        Tuple{Index{Vector{Pair{QN,Int64}}}},
        EmptyStorage{EmptyNumber,BlockSparse{EmptyNumber,Vector{EmptyNumber},1}},
      },
      Float64,
      Int64,
    },
  )   # time: 0.018442523
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.017981868
  @assert Base.precompile(
    Tuple{
      typeof(permute),
      ITensor,
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.017732373
  @assert Base.precompile(
    Tuple{
      typeof(replaceind),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      Index{Vector{Pair{QN,Int64}}},
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.016822455
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.016660046
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(eigen)),
      NamedTuple{
        (
          :ishermitian,
          :which_decomp,
          :tags,
          :maxdim,
          :mindim,
          :cutoff,
          :eigen_perturbation,
          :ortho,
          :normalize,
          :svd_alg,
        ),
        Tuple{Bool,Nothing,TagSet,Int64,Int64,Float64,Nothing,String,Bool,String},
      },
      typeof(eigen),
      ITensor,
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.013451855
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(combiner)),
      NamedTuple{(:dir, :tags),Tuple{Arrow,String}},
      typeof(combiner),
      Index{Vector{Pair{QN,Int64}}},
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.011506611
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      Float64,
      Pair{Index{Vector{Pair{QN,Int64}}},Int64},
      Pair{Index{Vector{Pair{QN,Int64}}},Int64},
    },
  )   # time: 0.011320483
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(combiner)),
      NamedTuple{(:dir, :tags),Tuple{Arrow,String}},
      typeof(combiner),
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.009607937
  @assert Base.precompile(
    Tuple{typeof(permute),ITensor,NTuple{4,Index{Vector{Pair{QN,Int64}}}}}
  )   # time: 0.00959132
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      DiagBlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        DiagBlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.008676937
  @assert Base.precompile(
    Tuple{
      typeof(replaceind!),
      ITensor,
      Index{Vector{Pair{QN,Int64}}},
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.008557502
  @assert Base.precompile(
    Tuple{
      typeof(==),
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00846187
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.007538412
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(svd)),
      NamedTuple{(:maxdim, :utags),Tuple{Int64,String}},
      typeof(svd),
      ITensor,
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.006647416
  @assert Base.precompile(Tuple{Type{TagSet},String})   # time: 0.006234448
  @assert Base.precompile(Tuple{typeof(sim),Vector{Index{Vector{Pair{QN,Int64}}}}})   # time: 0.006221086
  @assert Base.precompile(
    Tuple{
      typeof(permute),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.005509289
  @assert Base.precompile(
    Tuple{
      typeof(noprime),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.005396584
  @assert Base.precompile(
    Tuple{
      typeof(noprime),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.005214029
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds!),
      ITensor,
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.004601424
  @assert Base.precompile(Tuple{typeof(flux),ITensor,Block{2}})   # time: 0.004569341
  @assert Base.precompile(
    Tuple{
      typeof(settags),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      TagSet,
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.004546373
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        1,
        Tuple{Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},1},
      },
    },
  )   # time: 0.004459298
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      ITensor,
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.004386676
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(siteinds)),
      NamedTuple{(:conserve_qns,),Tuple{Bool}},
      typeof(siteinds),
      String,
      Int64,
    },
  )   # time: 0.004316439
  @assert Base.precompile(
    Tuple{
      typeof(replaceinds),
      ITensor,
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.004284349
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Vararg{Any,N} where N,
    },
  )   # time: 0.004172807
  @assert Base.precompile(
    Tuple{
      Type{Vector{IndexT} where IndexT<:Index},
      Index{Vector{Pair{QN,Int64}}},
      Vararg{Index{Vector{Pair{QN,Int64}}},N} where N,
    },
  )   # time: 0.003948658
  @assert Base.precompile(
    Tuple{
      typeof(_setindex!!),
      EmptyTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        EmptyStorage{Float64,BlockSparse{Float64,Vector{Float64},2}},
      },
      Float64,
      IndexVal{Index{Vector{Pair{QN,Int64}}}},
      IndexVal{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.003928109
  @assert Base.precompile(
    Tuple{
      typeof(_add),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.00389369
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        5,
        NTuple{5,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},5},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.003861998
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.003849556
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        5,
        NTuple{5,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},5},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.003845831
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        5,
        NTuple{5,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},5},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.003833467
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.003811549
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.003792728
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.003524243
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.003122797
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      BlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},2},
      },
    },
  )   # time: 0.003117909
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      Tensor{
        Number,
        3,
        Combiner,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
      },
    },
  )   # time: 0.003117105
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.003116956
  @assert Base.precompile(
    Tuple{
      typeof(prime),
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
    },
  )   # time: 0.003115171
  @assert Base.precompile(
    Tuple{
      typeof(hascommoninds),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.003056218
  @assert Base.precompile(
    Tuple{
      typeof(_contract),
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.002954446
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00295227
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.002892889
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(Type)),
      NamedTuple{(:tags,),Tuple{String}},
      Type{Index},
      Pair{QN,Int64},
      Pair{QN,Int64},
      Pair{QN,Int64},
    },
  )   # time: 0.002647182
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.002578369
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.002578002
  @assert Base.precompile(Tuple{Type{Sweeps},Int64})   # time: 0.002490389
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      Tensor{
        Number,2,Combiner,Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}}
      },
    },
  )   # time: 0.002479082
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Vararg{Any,N} where N,
    },
  )   # time: 0.002391966
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Vararg{
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        N,
      } where N,
    },
  )   # time: 0.00238876
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.002373397
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Vararg{NTuple{4,Index{Vector{Pair{QN,Int64}}}},N} where N,
    },
  )   # time: 0.002297418
  @assert Base.precompile(
    Tuple{
      typeof(dag),
      AllowAlias,
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
    },
  )   # time: 0.002132542
  @assert Base.precompile(
    Tuple{
      typeof(settags),
      DiagBlockSparseTensor{
        Float64,
        2,
        Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
        DiagBlockSparse{Float64,Vector{Float64},2},
      },
      TagSet,
      Index{Vector{Pair{QN,Int64}}},
    },
  )   # time: 0.002089967
  @assert Base.precompile(Tuple{typeof(isless),Pair{QN,Int64},Pair{QN,Int64}})   # time: 0.001830232
  @assert Base.precompile(Tuple{typeof(isint),SmallString})   # time: 0.001769182
  @assert Base.precompile(Tuple{typeof(dag),AllowAlias,ITensor})   # time: 0.001381586
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001364565
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.001339262
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.00132448
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.001310288
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.001291268
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001255944
  @assert Base.precompile(
    Tuple{
      typeof(_permute),
      NeverAlias,
      BlockSparseTensor{
        Float64,
        4,
        NTuple{4,Index{Vector{Pair{QN,Int64}}}},
        BlockSparse{Float64,Vector{Float64},4},
      },
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00125566
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{Index{Vector{Pair{QN,Int64}}},Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.001235331
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001206278
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001180919
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.001177626
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00113258
  @assert Base.precompile(
    Tuple{
      typeof(_intersect),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Vector{Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00112482
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff!),
      Vector{Index{Vector{Pair{QN,Int64}}}},
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001115587
  @assert Base.precompile(
    Tuple{
      typeof(_permute),
      NeverAlias,
      BlockSparseTensor{
        Float64,
        3,
        Tuple{
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
          Index{Vector{Pair{QN,Int64}}},
        },
        BlockSparse{Float64,Vector{Float64},3},
      },
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
    },
  )   # time: 0.001072638
  @assert Base.precompile(
    Tuple{
      typeof(_setdiff),
      Tuple{
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
        Index{Vector{Pair{QN,Int64}}},
      },
      NTuple{4,Index{Vector{Pair{QN,Int64}}}},
    },
  )   # time: 0.00106484

  ###################################################################################
  # NDTensors dmrg qn
  #

  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(svd)),
      NamedTuple{(:alg,),Tuple{String}},
      typeof(svd),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
    },
  )   # time: 1.3311685
  # XXX: error
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{2},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.6173956
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DiagTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Diag{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.3946228
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.25111172
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
    },
  )   # time: 0.18205081
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      DiagTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Diag{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.13669309
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
    },
  )   # time: 0.1107187
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
    },
  )   # time: 0.10938102
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._gemm!),
      Char,
      Char,
      Float64,
      Base.ReshapedArray{Float64,2,Vector{Float64},Tuple{}},
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
      Float64,
      Base.ReshapedArray{
        Float64,2,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Tuple{}
      },
    },
  )   # time: 0.10879537
  @assert Base.precompile(
    Tuple{
      typeof(copyto!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      DenseTensor{Float64,2,Tuple{Int64,Int64},Dense{Float64,Vector{Float64}}},
    },
  )   # time: 0.06301783
  @assert Base.precompile(
    Tuple{
      typeof(copy),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            Adjoint{Float64,Matrix{Float64}},
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.05344968
  @assert Base.precompile(
    Tuple{
      typeof(copyto!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            Adjoint{Float64,Matrix{Float64}},
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.026678292
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{2},
      Tuple{Int64,Int64},
      Block{2},
      Tuple{Int64,Int64},
      Val{4},
    },
  )   # time: 0.026060533
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      DenseTensor{Float64,2,Tuple{Int64,Int64},Dense{Float64,Vector{Float64}}},
      UnitRange{Int64},
      UnitRange{Int64},
    },
  )   # time: 0.025962748
  @assert Base.precompile(Tuple{typeof(copy),BlockSparse{Float64,Vector{Float64},4}})   # time: 0.023858083
  @assert Base.precompile(
    Tuple{typeof(NDTensors.combine_blocks),Vector{Block{2}},Int64,Vector{Int64}}
  )   # time: 0.022285657
  @assert Base.precompile(
    Tuple{typeof(NDTensors.combine_blocks),Vector{Block{3}},Int64,Vector{Int64}}
  )   # time: 0.02123815
  @assert Base.precompile(Tuple{typeof(sort),Tuple{Int64,Int64}})   # time: 0.020029295
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            Adjoint{Float64,Matrix{Float64}},
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
      UnitRange{Int64},
      UnitRange{Int64},
    },
  )   # time: 0.01732923
  @assert Base.precompile(
    Tuple{
      typeof(eigen),
      Hermitian{
        Float64,
        DenseTensor{
          Float64,
          2,
          Tuple{Int64,Int64},
          Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
        },
      },
    },
  )   # time: 0.01730679
  @assert Base.precompile(
    Tuple{
      typeof(copyto!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,2,Matrix{Float64},Tuple{UnitRange{Int64},UnitRange{Int64}},false
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.01667608
  @assert Base.precompile(Tuple{typeof(*),Float64,BlockSparse{Float64,Vector{Float64},3}})   # time: 0.012311829
  @assert Base.precompile(Tuple{typeof(norm),BlockSparse{Float64,Vector{Float64},3}})   # time: 0.01182627
  @assert Base.precompile(
    Tuple{
      typeof(copy),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,
              2,
              Base.ReshapedArray{
                Float64,
                2,
                Adjoint{Float64,Matrix{Float64}},
                Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
              },
              Tuple{UnitRange{Int64},UnitRange{Int64}},
              false,
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.011201464
  @assert Base.precompile(
    Tuple{
      typeof(copy),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,2,Matrix{Float64},Tuple{UnitRange{Int64},UnitRange{Int64}},false
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.010434925
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{3}},NTuple{5,Int64},Tuple{Int64,Int64}}
  )   # time: 0.007942336
  @assert Base.precompile(
    Tuple{
      typeof(copyto!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,
              2,
              Base.ReshapedArray{
                Float64,
                2,
                Adjoint{Float64,Matrix{Float64}},
                Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
              },
              Tuple{UnitRange{Int64},UnitRange{Int64}},
              false,
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
    },
  )   # time: 0.007778209
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{2},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.006501776
  @assert Base.precompile(Tuple{typeof(NDTensors.insertafter),Block{2},Tuple{Int64},Int64})   # time: 0.005869132
  @assert Base.precompile(Tuple{typeof(convert),Type{Block{2}},Tuple{UInt64,Int64}})   # time: 0.005855715
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        1,
        Tuple{Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64},
      Float64,
      Float64,
    },
  )   # time: 0.005832034
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.005645013
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.005631288
  @assert Base.precompile(
    Tuple{
      typeof(convert),
      Type{DiagTensor{Float64,2,Tuple{Int64,Int64},Diag{Float64,Vector{Float64}}}},
      DiagTensor{Float64,2,Tuple{Int64,Int64},Diag{Float64,Diag{Float64,Vector{Float64}}}},
    },
  )   # time: 0.004916287
  @assert Base.precompile(Tuple{typeof(nzblocks),BlockSparse{Float64,Vector{Float64},2}})   # time: 0.00482547
  @assert Base.precompile(Tuple{typeof(NDTensors.isblockless),Block{3},Block{3}})   # time: 0.004356135
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.004267705
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.004239671
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.004141683
  @assert Base.precompile(
    Tuple{typeof(NDTensors.insertat),Tuple{Int64,Int64,Int64},Tuple{Int64,Int64},Int64}
  )   # time: 0.004140997
  @assert Base.precompile(
    Tuple{typeof(NDTensors._permute_combdims),Tuple{Int64},Tuple{Int64,Int64}}
  )   # time: 0.004127886
  @assert Base.precompile(
    Tuple{typeof(NDTensors.perm_blocks),Vector{Block{2}},Int64,Vector{Int64}}
  )   # time: 0.0040685
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{3},
      Tuple{Int64,Int64,Int64},
      Block{4},
      NTuple{4,Int64},
      Val{3},
    },
  )   # time: 0.003858298
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{3},
      Tuple{Int64,Int64,Int64},
      Block{2},
      Tuple{Int64,Int64},
      Val{3},
    },
  )   # time: 0.003857063
  @assert Base.precompile(
    Tuple{typeof(NDTensors.perm_blocks),Vector{Block{3}},Int64,Vector{Int64}}
  )   # time: 0.003740321
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{2},
      Tuple{Int64,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{3},
    },
  )   # time: 0.003701912
  @assert Base.precompile(
    Tuple{typeof(tensor),Diag{Float64,Diag{Float64,Vector{Float64}}},Vararg{Any,N} where N}
  )   # time: 0.003601761
  @assert Base.precompile(Tuple{typeof(NDTensors.deleteat),Block{4},Tuple{Int64,Int64}})   # time: 0.003522632
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{5},
    },
  )   # time: 0.003430802
  @assert Base.precompile(Tuple{typeof(NDTensors.deleteat),Block{3},Tuple{Int64,Int64}})   # time: 0.003326959
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.003238392
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      BlockSparseTensor{Float64,0,Tuple{},BlockSparse{Float64,Vector{Float64},0}},
    },
  )   # time: 0.003114841
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.003097924
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.003063717
  @assert Base.precompile(
    Tuple{typeof(NDTensors.contract_labels),Type{Val{4}},NTuple{4,Int64},Tuple{Int64,Int64}}
  )   # time: 0.003017741
  @assert Base.precompile(
    Tuple{typeof(NDTensors.insertat),Tuple{Int64,Int64},Tuple{Int64,Int64},Int64}
  )   # time: 0.002960266
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{3},
      Tuple{Int64,Int64,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{4},
    },
  )   # time: 0.002893762
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{3},
      Tuple{Int64,Int64,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{2},
    },
  )   # time: 0.002722011
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002709946
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002613709
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002587505
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002504121
  @assert Base.precompile(Tuple{typeof(NDTensors.isblockless),Block{2},Block{2}})   # time: 0.002446814
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        0,
        Tuple{},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002441288
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002404394
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract!),
      DenseTensor{
        Float64,
        0,
        Tuple{},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.002383023
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{3},
      Tuple{Int64,Int64,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{0},
    },
  )   # time: 0.002361436
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{5},
      NTuple{5,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{4},
    },
  )   # time: 0.002353502
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{1},
      Tuple{Int64},
      Val{3},
    },
  )   # time: 0.002330633
  @assert Base.precompile(
    Tuple{
      Core.kwftype(typeof(NDTensors._truncated_blockdim)),
      NamedTuple{(:singular_values, :truncate),Tuple{Bool,Bool}},
      typeof(NDTensors._truncated_blockdim),
      DiagTensor{Float64,2,Tuple{Int64,Int64},Diag{Float64,Vector{Float64}}},
      Float64,
    },
  )   # time: 0.002322051
  @assert Base.precompile(Tuple{typeof(NDTensors.insertafter),Block{1},Tuple{Int64},Int64})   # time: 0.002311877
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{2},
      Tuple{Int64,Int64},
      Val{4},
    },
  )   # time: 0.002277473
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{3},
      Tuple{Int64,Int64,Int64},
      Val{3},
    },
  )   # time: 0.002238486
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{5},
      NTuple{5,Int64},
      Block{4},
      NTuple{4,Int64},
      Val{5},
    },
  )   # time: 0.002234501
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{5},
      NTuple{5,Int64},
      Block{2},
      Tuple{Int64,Int64},
      Val{3},
    },
  )   # time: 0.002221201
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{4},
      NTuple{4,Int64},
      Val{0},
    },
  )   # time: 0.002040847
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.contract_blocks),
      Block{4},
      NTuple{4,Int64},
      Block{4},
      NTuple{4,Int64},
      Val{4},
    },
  )   # time: 0.001963128
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001897563
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001893798
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001883682
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001865141
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001861894
  @assert Base.precompile(
    Tuple{
      typeof(convert),
      Type{
        DenseTensor{
          Float64,
          2,
          Tuple{Int64,Int64},
          Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
        },
      },
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
    },
  )   # time: 0.00185412
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      Tuple{Int64,Int64,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.00185278
  @assert Base.precompile(
    Tuple{typeof(NDTensors._permute_combdims),Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}}
  )   # time: 0.001851998
  @assert Base.precompile(
    Tuple{typeof(NDTensors._permute_combdims),Tuple{Int64,Int64},NTuple{4,Int64}}
  )   # time: 0.001829687
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors._contract_scalar_maybe_perm!),
      NDTensors.Order{1},
      DenseTensor{
        Float64,
        5,
        NTuple{5,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{5,Int64},
      DenseTensor{
        Float64,
        4,
        NTuple{4,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
      NTuple{4,Int64},
      Float64,
      Float64,
    },
  )   # time: 0.001781005
  @assert Base.precompile(
    Tuple{
      typeof(NDTensors.similar),
      Type{BlockSparseTensor{Float64,0,Tuple{},BlockSparse{Float64,Vector{Float64},0}}},
      NDTensors.Dictionaries.Dictionary{Block{0},Int64},
      Tuple{},
    },
  )   # time: 0.001562116
  @assert Base.precompile(
    Tuple{
      typeof(tensor),
      Diag{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      Tuple{Int64,Int64},
    },
  )   # time: 0.001536605
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,2,Matrix{Float64},Tuple{UnitRange{Int64},UnitRange{Int64}},false
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
      CartesianIndex{2},
    },
  )   # time: 0.001424076
  @assert Base.precompile(
    Tuple{
      typeof(array),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
    },
  )   # time: 0.001423944
  @assert Base.precompile(
    Tuple{
      typeof(array),
      DenseTensor{
        Float64,
        3,
        Tuple{Int64,Int64,Int64},
        Dense{Float64,SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}},
      },
    },
  )   # time: 0.001422581
  @assert Base.precompile(
    Tuple{typeof(NDTensors._permute_combdims),Tuple{Int64},Tuple{Int64,Int64,Int64}}
  )   # time: 0.001372653
  @assert Base.precompile(Tuple{typeof(NDTensors.deleteat),Block{2},Tuple{Int64}})   # time: 0.001151062
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            Adjoint{Float64,Matrix{Float64}},
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
      CartesianIndex{2},
    },
  )   # time: 0.001046506
  @assert Base.precompile(
    Tuple{
      typeof(getindex),
      DenseTensor{
        Float64,
        2,
        Tuple{Int64,Int64},
        Dense{
          Float64,
          Base.ReshapedArray{
            Float64,
            1,
            SubArray{
              Float64,
              2,
              Base.ReshapedArray{
                Float64,
                2,
                Adjoint{Float64,Matrix{Float64}},
                Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
              },
              Tuple{UnitRange{Int64},UnitRange{Int64}},
              false,
            },
            Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}},
          },
        },
      },
      CartesianIndex{2},
    },
  )
end
# COV_EXCL_STOP
