const CuDense{ElT,VecT}                 = Dense{ElT,VecT} where {VecT<:CuVector}
const CuDenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:CuDense}

Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:CuArray, SB<:Array} = Dense{T, SA}(CuArray(x))
Dense{T, SA}(x::Dense{T, SB}) where {T<:Number, SA<:Array, SB<:CuArray} = Dense{T, SA}(collect(x.data))
Dense{T, S}(size::Integer) where {T, S<:CuArray{<:T}} = Dense{T, S}(CUDA.zeros(T, size))
function Dense{T, S}(x::T, size::Integer) where {T, S<:CuArray{<:T}}
    arr = CuArray{T}(undef, size)
    fill!(arr, x)
    Dense{T, S}(arr)
end
cpu(x::CuDense{T}) where {T<:Number} = Dense(collect(x.data))
Base.complex(::Type{Dense{ElT, VT}}) where {ElT, VT<:CuArray} = Dense{complex(ElT),CuVector{complex(ElT)}}

similartype(::Type{<:CuArray{<:Any,N}}, eltype::Type) where {N} = CuArray{eltype,N}
NDTensors.similar(::Type{<:CuArray{T}}, dims) where {T} = CuArray{T,length(dims)}(undef, dims)

CuArray(x::CuDense{ElT}) where {ElT} = CuVector{ElT}(data(x))
CuArray{ElT, N}(x::CuDenseTensor{ElT, N}) where {ElT, N} = CuArray{ElT, N}(reshape(data(store(x)), dims(inds(x))...))
CuArray(x::CuDenseTensor{ElT, N}) where {ElT, N} = CuArray{ElT, N}(x)

*(D::Dense{T, AT},x::S) where {T,AT<:CuArray,S<:Number} = Dense(x .* data(D))

Base.:(==)(::Type{<:CuDense{ElT1,CVec1}}, ::Type{<:CuDense{ElT2,CVec2}}) where {ElT1,ElT2,CVec1,CVec2} = (ElT1 == ElT2)
Base.getindex(D::CuDense{<:Number})          = collect(data(D))[]
Base.getindex(D::CuDenseTensor{<:Number, 0}) = store(D)[]
LinearAlgebra.norm(T::CuDenseTensor) = norm(data(store(T)))

# This is for type promotion for Scalar*Dense
function Base.promote_rule(::Type{<:Dense{ElT1,CuVector{ElT1}}},
                           ::Type{ElT2}) where {ElT1,
                                                ElT2<:Number}
  ElR  = promote_type(ElT1,ElT2)
  VecR = CuVector{ElR}
  return Dense{ElR,VecR}
end

function Base.permutedims(T::CuDenseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  Tp = NDTensors.similar(T,ITensors.NDTensors.permute(inds(T),perm))
  #Tp = permute(T,perm; always_copy=true)
  permute!(Tp, T)
  return Tp
end

function Base.permutedims!(R::CuDenseTensor{<:Number,N},
                           T::CuDenseTensor{<:Number,N},
                           perm::NTuple{N,Int}) where {N}
    return permutedims!!(R, T, perm)
end

function permutedims!!(B::Tensor{ElT,N,StoreT,IndsB},
                       A::Tensor{ElT,N,StoreT,IndsA},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->permute!(r,t)) where {N,ElT,IndsB,IndsA,StoreT<:CuDense{ElT}}
  Ais = inds(A)
  Bis = ITensors.NDTensors.permute(inds(A), perm)
  B = f(B, A)
  return B
end

function Base.similar(::Type{<:CuDenseTensor{ElT}},
                      inds::Dims) where {ElT}
    storage_arr = CuVector{ElT}(undef,dim(inds)) 
    return Tensor(Dense(storage_arr),inds)
end

function Base.similar(::Type{<:CuDenseTensor{ElT}},
                      inds) where {ElT}
    storage_arr = CuVector{ElT}(undef,dim(inds)) 
    return Tensor(Dense(storage_arr),inds)
end

function outer!(R::CuDenseTensor,
                T1::CuDenseTensor,
                T2::CuDenseTensor)
  R_dat = vec(array(T1))*transpose(vec(array(T2)))
  copyto!(data(store(R)), vec(R_dat)) 
  inds_outer = unioninds(inds(T1),inds(T2))
  return R
end

function contract!!(R::CuDenseTensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::CuDenseTensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::CuDenseTensor{<:Number,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  if N1==0
    # TODO: replace with an add! function?
    # What about doing `R .= T1[] .* PermutedDimsArray(T2,perm)`?
    perm = getperm(labelsR,labelsT2)
    newT2 = Tensor(Dense(data(store(T1)).*data(store(T2))), inds(T2))
    permute!(R,newT2)
  elseif N2==0
    perm = getperm(labelsR,labelsT1)
    newT1 = Tensor(Dense(data(store(T2)).*data(store(T1))), inds(T1))
    permute!(R,newT1)
  elseif N1+N2==NR
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    # TODO: implement the in-place version directly
    R = outer!!(R,T1,T2)
    inds_outer = unioninds(inds(T1),inds(T2))
    R = Tensor(store(R), inds_outer)
  else
    R = _contract!!(R,labelsR,T1,labelsT1,T2,labelsT2)
  end
  return R
end

function permutedims!!(B::CuDenseTensor{ElT,0},
                       A::CuDenseTensor{ElT,0},
                       perm::NTuple{0,Int},
                       f=(r,t)->permute!(r,t)) where {ElT<:Number}
    Cs = f(B, A)
    return Tensor(Dense(vec(Cs)), IndexSet{0}()) 
end

function permutedims!!(B::CuDenseTensor{ElT,N},
                       A::CuDenseTensor{ElT,0},
                       perm::NTuple{N,Int},
                       f=(r,t)->permute!(r,t)) where {N, ElT<:Number}
    Cis = ITensors.NDTensors.permute(inds(B), perm)
    Cs = f(B, A)
    return Tensor(Dense(vec(Cs)), Cis) 
end

function _contract_scalar!(R::CuDenseTensor{ElR}, labelsR,
                           T1::Number, labelsT1,
                           T2::Number, labelsT2,
                           α = one(ElR), β = zero(ElR)) where {ElR}
  if iszero(β)
    copyto!(data(R), [α * T1 * T2])
  elseif iszero(α)
    copyto!(data(R), β.*data(R))
  else
    copyto!(data(R), [α * T1 * T2] .+ β.*data(R))
  end
  return R
end

function _contract_scalar!(R::CuDenseTensor{ElR,NR}, labelsR,
                           T₁::CuDenseTensor, labelsT₁,
                           T₂::CuDenseTensor, labelsT₂,
                           α = one(ElR), β=zero(ElR)) where {ElR,NR}
    if nnz(T₁) == nnz(T₂) == 1
        new_R = Tensor(Dense(data(store(T₁)).*data(store(T₂))), inds(R))
        copyto!(store(R), store(new_R))
    elseif nnz(T₁) == 1
        props = ContractionProperties(labelsT₁, labelsT₂, labelsR)
        compute_contraction_properties!(props, T₁, T₂, R)
        R = _contract!(R, T₁, T₂, props, α, β)
        #perm = getperm(labelsR,labelsT₂)
        #newT2 = Tensor(Dense(data(store(T₁)).*data(store(T₂))), inds(T₂))
        #permute!(R,newT2)
    elseif nnz(T₂) == 1
        props = ContractionProperties(labelsT₁, labelsT₂, labelsR)
        compute_contraction_properties!(props, T₁, T₂, R)
        R = _contract!(R, T₁, T₂, props, α, β)
        #perm = getperm(labelsR,labelsT₁)
        #newT1 = Tensor(Dense(data(store(T₁)).*data(store(T₂))), inds(T₁))
        #permute!(R,newT1)
    else
        error("In _contract_scalar!, one tensor must be a scalar")
    end
    return R
end

function _gemm_contract!(CT::DenseTensor{El,NC},
                         AT::DenseTensor{El,NA},
                         BT::DenseTensor{El,NB},
                         props::ContractionProperties,
                         α::Number=one(El),
                         β::Number=zero(El)) where {El,NC,NA,NB}
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
            AM = reshape(A,props.dmid,props.dleft)
            tA = 'T'
        else
            AM = reshape(A,props.dleft,props.dmid)
        end
    end

    tB = 'N'
    if props.permuteB
        pB = NTuple{NB,Int}(props.PB)
        Bp = permutedims(B, pB)
        BM = reshape(Bp, props.dmid, props.dright)
    else
        if Btrans(props)
            BM = reshape(B,props.dright,props.dmid)
            tB = 'T'
        else
            BM = reshape(B,props.dmid,props.dright)
        end
    end

    #TODO: this logic may be wrong
    if props.permuteC
        #Need to copy here since we will be permuting
        #into C later
        CM = reshape(copy(C), props.dleft, props.dright)
    else
        if Ctrans(props)
            CM = reshape(C,props.dleft,props.dright)
            (AM,BM) = (BM,AM)
            if tA==tB
                tA = tB = (tA == 'T' ? 'N' : 'T')
            end
        else
            CM = reshape(C, props.dleft, props.dright)
        end
    end

    CM = CUBLAS.gemm!(tA,tB,El(α),AM,BM,El(β),CM)

    if props.permuteC
        pC = NTuple{NC,Int}(props.PC)
        Cr = reshape(CM,props.newCrange...)
        @strided C .= permutedims(Cr, pC)
    end
    return C
end

function _contract!(CT::CuDenseTensor{El,NC},
                    AT::CuDenseTensor{El,NA},
                    BT::CuDenseTensor{El,NB},
                    props::ContractionProperties,
                    α::Number=one(El),β::Number=zero(El)) where {El,NC,NA,NB}
  if ndims(CT) > 12 || ndims(BT) > 12 || ndims(AT) > 12
    return _gemm_contract!(CT, AT, BT, props, α, β)
  end
  Ainds = inds(AT)
  Adims = dims(Ainds)
  Binds = inds(BT)
  Bdims = dims(Binds)
  Cinds = inds(CT)
  Cdims = dims(Cinds)
  Adata = reshape(data(store(AT)),Adims...)
  Bdata = reshape(data(store(BT)),Bdims...)
  Cdata = reshape(data(store(CT)),Cdims...)
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
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Binds)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  for (ii, ic) in enumerate(Cinds)
      ctcinds[ii] = findfirst(x->x==ic, ind_dict)
  end
  id_op    = CUDA.CUTENSOR.CUTENSOR.CUTENSOR_OP_IDENTITY
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
          algo  = dict_val
          #plan  = dict_val[2]
          Cdata = CUDA.CUTENSOR.contraction!(α, Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, β, Cdata, Vector{Char}(ctcinds), id_op, id_op; algo=algo)
      else
          # loop through all algos
          # pick the fastest one
          # store that plan!
          best_time = 1e6
          best_plan = nothing
          best_algo = nothing
          max_algos = Ref{Int32}(C_NULL)
          CUDA.CUTENSOR.cutensorContractionMaxAlgos(max_algos)
          # fix once the other options are documented
          #algos = collect(Cint(CUDA.CUTENSOR.CUTENSOR_ALGO_GETT):Cint(max_algos[] - 1))
          algos = collect(Cint(CUDA.CUTENSOR.CUTENSOR_ALGO_GETT):Cint(-1))
          for algo in reverse(algos)
              try
                  Cdata, this_time, bytes, gctime, memallocs = @timed CUDA.CUTENSOR.contraction!(α, Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, β, Cdata, Vector{Char}(ctcinds), id_op, id_op; algo=CUDA.CUTENSOR.cutensorAlgo_t(algo))
                  if this_time < best_time
                      best_time = this_time
                      #best_plan = this_plan
                      best_algo = CUDA.CUTENSOR.cutensorAlgo_t(algo)
                  end
              catch err
                  @warn "Algorithm $algo not supported"
              end
          end
          ContractionPlans[dict_key] = best_algo
      end
  else
      Cdata = CUDA.CUTENSOR.contraction!(α, Adata, Vector{Char}(ctainds), id_op, Bdata, Vector{Char}(ctbinds), id_op, β, Cdata, Vector{Char}(ctcinds), id_op, id_op)
  end
  return parent(Cdata)
end

function Base.:+(B::CuDenseTensor, A::CuDenseTensor)
  opC  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opA  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opAC = CUTENSOR.CUTENSOR_OP_ADD
  Ais = inds(A)
  Bis = inds(B)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(inds(A))
      push!(ind_dict, i)
  end
  Adata = data(store(A))
  Bdata = data(store(B))
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  CUTENSOR.elementwiseBinary!(one(eltype(Adata)), reshapeAdata, ctainds, opA, one(eltype(Bdata)), reshapeBdata, ctbinds, opC, C, ctcinds, opAC)
  copyto!(data(store(B)), vec(C))
  return B
end

function Base.:+(B::CuDense, Bis::IndexSet, A::CuDense, Ais::IndexSet)
  opA  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opC  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opAC = CUTENSOR.CUTENSOR_OP_ADD
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  Cis = Bis
  C = CUTENSOR.elementwiseBinary!(1, reshapeAdata, ctainds, opA, 1, reshapeBdata, ctbinds, opC, C, ctcinds, opAC)
  copyto!(data(B), vec(C))
  return C
end

function Base.:-(B::CuDenseTensor, A::CuDenseTensor)
  opC  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opA  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opAC = CUTENSOR.CUTENSOR_OP_ADD
  Ais = inds(A)
  Bis = inds(B)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(inds(A))
      push!(ind_dict, i)
  end
  Adata = data(store(A))
  Bdata = data(store(B))
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis))
  CUTENSOR.elementwiseBinary!(one(eltype(Adata)), reshapeAdata, ctainds, opA, -one(eltype(Bdata)), reshapeBdata, ctbinds, opC, C, ctcinds, opAC)
  copyto!(data(store(B)), vec(C))
  return B
end

function Base.:-(A::CuDense, Ais::IndexSet, B::CuDense, Bis::IndexSet)
  opA  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opC  = CUTENSOR.CUTENSOR_OP_IDENTITY
  opAC = CUTENSOR.CUTENSOR_OP_ADD
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  ctcinds = copy(ctbinds)
  C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
  Cis = Bis
  C = CUTENSOR.elementwiseBinary!(one(eltype(Adata)), reshapeAdata, ctainds, opA, -one(eltype(Bdata)), reshapeBdata, ctbinds, opC, C, ctcinds, opAC)
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
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  if ndims(A) < 12 # use CUTENSOR
      ctainds = zeros(Int, length(Ais))
      ctbinds = zeros(Int, length(Bis))
      for (ii, ia) in enumerate(Ais)
          ctainds[ii] = findfirst(x->x==ia, ind_dict)
      end
      for (ii, ib) in enumerate(Bis)
          ctbinds[ii] = findfirst(x->x==ib, ind_dict)
      end
      CUDA.CUTENSOR.permutation!(one(eltype(Adata)), reshapeAdata, Vector{Char}(ctainds), reshapeBdata, Vector{Char}(ctbinds))
  else # use GPUArrays
      perm = Int[]
      for aix in Ais
        b_pos = findfirst(bix->bix==aix, Bis)
        push!(perm, b_pos)
      end
      @assert isperm(perm)
      permutedims!(reshapeBdata, reshapeAdata, invperm(perm))
  end
  return vec(reshapeBdata) 
end

function Base.permute!(B::CuDense, Bis::IndexSet, A::CuDense, Ais::IndexSet)
  ind_dict = Vector{Index}()
  for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
  end
  Adata = data(A)
  Bdata = data(B)
  reshapeBdata = reshape(Bdata,dims(Bis)...)
  reshapeAdata = reshape(Adata,dims(Ais)...)
  ctainds = zeros(Int, length(Ais))
  ctbinds = zeros(Int, length(Bis))
  for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x->x==ia, ind_dict)
  end
  for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x->x==ib, ind_dict)
  end
  
  CUDA.CUTENSOR.permutation!(one(eltype(Adata)), reshapeAdata, Vector{Char}(ctainds), reshapeBdata, Vector{Char}(ctbinds)) 
  return vec(reshapeBdata) 
end

Base.:/(A::CuDenseTensor, x::Number) = A * inv(x)
