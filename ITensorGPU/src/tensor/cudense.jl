using LinearAlgebra: BlasFloat

const CuDense{ElT,VecT} = Dense{ElT,VecT} where {VecT<:CuVector}
const CuDenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:CuDense}

# function Dense{T,SA}(x::Dense{T,SB}) where {T<:Number,SA<:CuArray,SB<:Array}
#   return Dense{T,SA}(CuArray(x))
# end
# function Dense{T,SA}(x::Dense{T,SB}) where {T<:Number,SA<:Array,SB<:CuArray}
#   return Dense{T,SA}(collect(x.data))
# end
# Dense{T,S}(size::Integer) where {T,S<:CuArray{<:T}} = Dense{T,S}(CUDA.zeros(T, size))
# function Dense{T,S}(x::T, size::Integer) where {T,S<:CuArray{<:T}}
#   arr = CuArray{T}(undef, size)
#   fill!(arr, x)
#   return Dense{T,S}(arr)
# end

function Base.complex(::Type{Dense{ElT,VT}}) where {ElT,VT<:CuArray}
  return Dense{complex(ElT),CuVector{complex(ElT)}}
end

CuArray(x::CuDense{ElT}) where {ElT} = CuVector{ElT}(data(x))
function CuArray{ElT,N}(x::CuDenseTensor{ElT,N}) where {ElT,N}
  return CuArray{ElT,N}(reshape(data(store(x)), dims(inds(x))...))
end
CuArray(x::CuDenseTensor{ElT,N}) where {ElT,N} = CuArray{ElT,N}(x)

*(D::Dense{T,AT}, x::S) where {T,AT<:CuArray,S<:Number} = Dense(x .* data(D))

Base.getindex(D::CuDense{<:Number}) = collect(data(D))[]
Base.getindex(D::CuDenseTensor{<:Number,0}) = store(D)[]
LinearAlgebra.norm(T::CuDenseTensor) = norm(data(store(T)))

function Base.copyto!(R::CuDenseTensor{<:Number,N}, T::CuDenseTensor{<:Number,N}) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end

# This is for type promotion for Scalar*Dense
function Base.promote_rule(
  ::Type{<:Dense{ElT1,CuVector{ElT1}}}, ::Type{ElT2}
) where {ElT1,ElT2<:Number}
  ElR = promote_type(ElT1, ElT2)
  VecR = CuVector{ElR}
  return Dense{ElR,VecR}
end

function permutedims!!(
  B::Tensor{ElT,N,StoreT,IndsB},
  A::Tensor{ElT,N,StoreT,IndsA},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> permute!(r, t),
) where {N,ElT,IndsB,IndsA,StoreT<:CuDense{ElT}}
  Ais = inds(A)
  Bis = ITensors.NDTensors.permute(inds(A), perm)
  B = f(B, A)
  return B
end

import ITensors.NDTensors: GemmBackend, auto_select_backend, _gemm!
function backend_cutensor()
  return gemm_backend[] = :CUTENSOR
end
function backend_cublas()
  return gemm_backend[] = :CUBLAS
end

@inline function auto_select_backend(
  ::Type{<:CuArray{<:BlasFloat}},
  ::Type{<:CuArray{<:BlasFloat}},
  ::Type{<:CuArray{<:BlasFloat}},
)
  return GemmBackend(:CUBLAS)
end

@inline function auto_select_backend(
  ::Type{<:CuArray{<:BlasFloat}}, ::Type{<:CuArray{<:BlasFloat}}, ::Type{<:AbstractVecOrMat}
)
  return GemmBackend(:GenericCUDA)
end

# CUBLAS matmul
function _gemm!(
  ::GemmBackend{:CUBLAS},
  tA,
  tB,
  alpha,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta,
  C::AbstractVecOrMat,
)
  return CUBLAS.gemm!(tA, tB, alpha, A, B, beta, C)
end

# CUDA generic matmul
function _gemm!(
  ::GemmBackend{:GenericCUDA},
  tA,
  tB,
  alpha,
  A::AbstractVecOrMat,
  B::AbstractVecOrMat,
  beta,
  C::CuDenseTensor,
)
  C_dat = reshape(data(store(C)), size(C))
  A_ = tA == 'T' ? transpose(A) : A
  B_ = tB == 'T' ? transpose(B) : B
  C_dat = mul!(C_dat, A_, B_, alpha, beta)
  copyto!(data(store(C)), C_dat)
  return C
end

function _contract_scalar!(
  R::CuDenseTensor{ElR,NR},
  labelsR,
  T₁::CuDenseTensor,
  labelsT₁,
  T₂::CuDenseTensor,
  labelsT₂,
  α=one(ElR),
  β=zero(ElR),
) where {ElR,NR}
  if nnz(T₁) == nnz(T₂) == 1
    new_R = Tensor(Dense(data(store(T₁)) .* data(store(T₂))), inds(R))
    copyto!(store(R), store(new_R))
  elseif nnz(T₁) == 1
    props = ContractionProperties(labelsT₁, labelsT₂, labelsR)
    compute_contraction_properties!(props, T₁, T₂, R)
    R = _contract!(R, T₁, T₂, props, α, β)
  elseif nnz(T₂) == 1
    props = ContractionProperties(labelsT₁, labelsT₂, labelsR)
    compute_contraction_properties!(props, T₁, T₂, R)
    R = _contract!(R, T₁, T₂, props, α, β)
  else
    error("In _contract_scalar!, one tensor must be a scalar")
  end
  return R
end

function _gemm_contract!(
  CT::DenseTensor{El,NC},
  AT::DenseTensor{El,NA},
  BT::DenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  # TODO: directly use Tensor instead of Array
  C = array(CT)
  A = array(AT)
  B = array(BT)

  tA = 'N'
  if props.permuteA
    pA = NTuple{NA,Int}(props.PA)
    Ap = permutedims(A, pA)
    AM = reshape(Ap, props.dmid, props.dleft)
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(props)
      AM = reshape(A, props.dmid, props.dleft)
      tA = 'T'
    else
      AM = reshape(A, props.dleft, props.dmid)
    end
  end

  tB = 'N'
  if props.permuteB
    pB = NTuple{NB,Int}(props.PB)
    Bp = permutedims(B, pB)
    BM = reshape(Bp, props.dmid, props.dright)
  else
    if Btrans(props)
      BM = reshape(B, props.dright, props.dmid)
      tB = 'T'
    else
      BM = reshape(B, props.dmid, props.dright)
    end
  end

  #TODO: this logic may be wrong
  if props.permuteC
    #Need to copy here since we will be permuting
    #into C later
    CM = reshape(copy(C), props.dleft, props.dright)
  else
    if Ctrans(props)
      CM = reshape(C, props.dleft, props.dright)
      (AM, BM) = (BM, AM)
      if tA == tB
        tA = tB = (tA == 'T' ? 'N' : 'T')
      end
    else
      CM = reshape(C, props.dleft, props.dright)
    end
  end

  CM = CUBLAS.gemm!(tA, tB, El(α), AM, BM, El(β), CM)

  if props.permuteC
    pC = NTuple{NC,Int}(props.PC)
    Cr = reshape(CM, props.newCrange...)
    @strided C .= permutedims(Cr, pC)
  end
  return C
end

function _contract!(
  CT::CuDenseTensor{El,NC},
  AT::CuDenseTensor{El,NA},
  BT::CuDenseTensor{El,NB},
  props::ContractionProperties,
  α::Number=one(El),
  β::Number=zero(El),
) where {El,NC,NA,NB}
  if ndims(CT) > 12 || ndims(BT) > 12 || ndims(AT) > 12
    return _gemm_contract!(CT, AT, BT, props, α, β)
  end
  Ainds = inds(AT)
  Adims = dims(Ainds)
  Binds = inds(BT)
  Bdims = dims(Binds)
  Cinds = inds(CT)
  Cdims = dims(Cinds)
  Adata = reshape(data(store(AT)), Adims...)
  Bdata = reshape(data(store(BT)), Bdims...)
  Cdata = reshape(data(store(CT)), Cdims...)
  contracted = commoninds(Ainds, Binds)
  A_only = uniqueinds(Ainds, Binds)
  B_only = uniqueinds(Binds, Ainds)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(contracted)
    push!(ind_dict, i)
  end
  if length(A_only) > 0
    for (idx, i) in enumerate(A_only)
      push!(ind_dict, i)
    end
  end
  if length(B_only) > 0
    for (idx, i) in enumerate(B_only)
      push!(ind_dict, i)
    end
  end
  ctainds = zeros(Int, length(Ainds))
  ctbinds = zeros(Int, length(Binds))
  ctcinds = zeros(Int, length(Cinds))
  for (ii, ia) in enumerate(Ainds)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Binds)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  for (ii, ic) in enumerate(Cinds)
    ctcinds[ii] = findfirst(x -> x == ic, ind_dict)
  end
  id_op = cuTENSOR.CUTENSOR_OP_IDENTITY
  dict_key = ""
  for cc in zip(ctcinds, Cdims)
    dict_key *= string(cc[1]) * "," * string(cc[2]) * ","
  end
  for aa in zip(ctainds, Adims)
    dict_key *= string(aa[1]) * "," * string(aa[2]) * ","
  end
  for bb in zip(ctbinds, Bdims)
    dict_key *= string(bb[1]) * "," * string(bb[2]) * ","
  end
  if haskey(ENV, "CUTENSOR_AUTOTUNE") && tryparse(Int, ENV["CUTENSOR_AUTOTUNE"]) == 1
    if haskey(ContractionPlans, dict_key)
      dict_val = ContractionPlans[dict_key]
      algo = dict_val
      #plan  = dict_val[2]
      Cdata = cuTENSOR.contraction!(
        α,
        Adata,
        Vector{Char}(ctainds),
        id_op,
        Bdata,
        Vector{Char}(ctbinds),
        id_op,
        β,
        Cdata,
        Vector{Char}(ctcinds),
        id_op,
        id_op;
        algo=algo,
      )
    else
      # loop through all algos
      # pick the fastest one
      # store that plan!
      best_time = 1e6
      best_plan = nothing
      best_algo = nothing
      max_algos = Ref{Int32}(C_NULL)
      cuTENSOR.cutensorContractionMaxAlgos(max_algos)
      # fix once the other options are documented
      #algos = collect(Cint(cuTENSOR.CUTENSOR_ALGO_GETT):Cint(max_algos[] - 1))
      algos = collect(Cint(cuTENSOR.CUTENSOR_ALGO_GETT):Cint(-1))
      for algo in reverse(algos)
        try
          Cdata, this_time, bytes, gctime, memallocs = @timed cuTENSOR.contraction!(
            α,
            Adata,
            Vector{Char}(ctainds),
            id_op,
            Bdata,
            Vector{Char}(ctbinds),
            id_op,
            β,
            Cdata,
            Vector{Char}(ctcinds),
            id_op,
            id_op;
            algo=cuTENSOR.cutensorAlgo_t(algo),
          )
          if this_time < best_time
            best_time = this_time
            #best_plan = this_plan
            best_algo = cuTENSOR.cutensorAlgo_t(algo)
          end
        catch err
          @warn "Algorithm $algo not supported"
        end
      end
      ContractionPlans[dict_key] = best_algo
    end
  else
    Cdata = cuTENSOR.contraction!(
      α,
      Adata,
      Vector{Char}(ctainds),
      id_op,
      Bdata,
      Vector{Char}(ctbinds),
      id_op,
      β,
      Cdata,
      Vector{Char}(ctcinds),
      id_op,
      id_op,
    )
  end
  return parent(Cdata)
end

function Base.:+(B::CuDenseTensor, A::CuDenseTensor)
  opC = cuTENSOR.CUTENSOR_OP_IDENTITY
  opA = cuTENSOR.CUTENSOR_OP_IDENTITY
  opAC = cuTENSOR.CUTENSOR_OP_ADD
  Ais = inds(A)
  Bis = inds(B)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(inds(A))
    push!(ind_dict, i)
  end
  Adata = data(store(A))
  Bdata = data(store(B))
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  cuTENSOR.elementwiseBinary!(
    one(eltype(Adata)),
    reshapeAdata,
    ctainds,
    opA,
    one(eltype(Bdata)),
    reshapeBdata,
    ctbinds,
    opC,
    C,
    ctcinds,
    opAC,
  )
  copyto!(data(store(B)), vec(C))
  return B
end

function Base.:+(B::CuDense, Bis::IndexSet, A::CuDense, Ais::IndexSet)
  opA = cuTENSOR.CUTENSOR_OP_IDENTITY
  opC = cuTENSOR.CUTENSOR_OP_IDENTITY
  opAC = cuTENSOR.CUTENSOR_OP_ADD
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
    push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  Cis = Bis
  C = cuTENSOR.elementwiseBinary!(
    1, reshapeAdata, ctainds, opA, 1, reshapeBdata, ctbinds, opC, C, ctcinds, opAC
  )
  copyto!(data(B), vec(C))
  return C
end

function Base.:-(B::CuDenseTensor, A::CuDenseTensor)
  opC = cuTENSOR.CUTENSOR_OP_IDENTITY
  opA = cuTENSOR.CUTENSOR_OP_IDENTITY
  opAC = cuTENSOR.CUTENSOR_OP_ADD
  Ais = inds(A)
  Bis = inds(B)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(inds(A))
    push!(ind_dict, i)
  end
  Adata = data(store(A))
  Bdata = data(store(B))
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis))
  cuTENSOR.elementwiseBinary!(
    -one(eltype(Adata)),
    reshapeAdata,
    ctainds,
    opA,
    one(eltype(Bdata)),
    reshapeBdata,
    ctbinds,
    opC,
    C,
    ctcinds,
    opAC,
  )
  copyto!(data(store(B)), vec(C))
  return B
end

function Base.:-(A::CuDense, Ais::IndexSet, B::CuDense, Bis::IndexSet)
  opA = cuTENSOR.CUTENSOR_OP_IDENTITY
  opC = cuTENSOR.CUTENSOR_OP_IDENTITY
  opAC = cuTENSOR.CUTENSOR_OP_ADD
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
    push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  Cis = Bis
  C = cuTENSOR.elementwiseBinary!(
    one(eltype(Adata)),
    reshapeAdata,
    ctainds,
    opA,
    -one(eltype(Bdata)),
    reshapeBdata,
    ctbinds,
    opC,
    C,
    ctcinds,
    opAC,
  )
  copyto!(data(B), vec(C))
  return C
end

function Base.permute!(B::CuDenseTensor, A::CuDenseTensor)
  Ais = inds(A)
  Bis = inds(B)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
    push!(ind_dict, i)
  end
  Adata = data(store(A))
  Bdata = data(store(B))
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  if ndims(A) < 40 # use CUTENSOR
    ctainds = zeros(Int, length(Ais))
    ctbinds = zeros(Int, length(Bis))
    for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x -> x == ia, ind_dict)
    end
    for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
    end
    cuTENSOR.permutation!(
      one(eltype(Adata)),
      reshapeAdata,
      Vector{Char}(ctainds),
      reshapeBdata,
      Vector{Char}(ctbinds),
    )
  else # use GPUArrays
    perm = Int[]
    for aix in Ais
      b_pos = findfirst(bix -> bix == aix, Bis)
      push!(perm, b_pos)
    end
    @assert isperm(perm)
    permutedims!(reshapeBdata, reshapeAdata, invperm(perm))
  end
  return Tensor(Dense(vec(reshapeBdata)), inds(B))
end

function Base.permute!(B::CuDense, Bis::IndexSet, A::CuDense, Ais::IndexSet)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
    push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata, dims(Bis)...)
  reshapeAdata = reshape(Adata, dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
    ctainds[ii] = findfirst(x -> x == ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
    ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
  end

  cuTENSOR.permutation!(
    one(eltype(Adata)),
    reshapeAdata,
    Vector{Char}(ctainds),
    reshapeBdata,
    Vector{Char}(ctbinds),
  )
  return Tensor(Dense(reshapeBdata), Tuple(Bis))
end

Base.:/(A::CuDenseTensor, x::Number) = A * inv(x)
