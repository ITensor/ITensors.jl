export Dense

#
# Dense storage
#

struct Dense{T} <: TensorStorage
  data::Vector{T}
  Dense{T}(data) where {T} = new{T}(convert(Vector{T},data))
  Dense{T}() where {T} = new{T}(Vector{T}())
end

data(D::Dense) = D.data

# Convenient functions for Dense storage type
Base.getindex(D::Dense,i::Integer) = data(D)[i]
Base.setindex!(D::Dense,v,i::Integer) = (data(D)[i] = v)

Base.similar(D::Dense{T}) where {T} = Dense{T}(similar(data(D)))
Base.similar(D::Dense{T},dims) where {T} = Dense{T}(similar(data(D),dim(dims)))
Base.similar(::Type{Dense{T}},dims) where {T} = Dense{T}(similar(Vector{T},dim(dims)))
Base.similar(D::Dense,::Type{T}) where {T} = Dense{T}(similar(data(D),T))
Base.copy(D::Dense{T}) where {T} = Dense{T}(copy(data(D)))

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::Dense{T}) where {T} = Dense{complex(T)}(complex(data(D)))

Base.eltype(::Dense{T}) where {T} = eltype(T)
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::Dense{Nothing}) = Nothing
Base.eltype(::Type{Dense{T}}) where {T} = eltype(T)

Base.promote_rule(::Type{Dense{T1}},::Type{Dense{T2}}) where {T1,T2} = Dense{promote_type(T1,T2)}
Base.convert(::Type{Dense{R}},D::Dense) where {R} = Dense{R}(convert(Vector{R},data(D)))

#
# DenseTensor (Tensor using Dense storage)
#

const DenseTensor{El,N,Inds} = Tensor{El,N,<:Dense,Inds}

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{TensorT}) where {TensorT<:DenseTensor} = IndexLinear()
Base.getindex(T::DenseTensor,i::Integer) = store(T)[i]
Base.setindex!(T::DenseTensor,v,i::Integer) = (store(T)[i] = v)

Base.strides(T::DenseTensor) = strides(inds(T))

# Needed for passing Tensor{T,2} to BLAS
function Base.unsafe_convert(::Type{Ptr{ElT}},T::DenseTensor{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT},data(store(T)))
end

function Base.convert(::Type{TensorR},
                      T::DenseTensor{ElT,N}) where {TensorR<:Tensor{ElR,N,StoreR}} where 
                                                   {ElR,ElT,N,StoreR<:Dense}
  return Tensor(convert(StoreR,store(T)),inds(T))
end

# Reshape a DenseTensor using the specified dimensions
function Base.reshape(T::DenseTensor,dims)
  dim(T)==dim(dims) || error("Total new dimension must be the same as the old dimension")
  return Tensor(store(T),dims)
end
# This version fixes method ambiguity with AbstractArray reshape
function Base.reshape(T::DenseTensor,dims::Dims)
  dim(T)==dim(dims) || error("Total new dimension must be the same as the old dimension")
  return Tensor(store(T),dims)
end

# Create an Array that is a view of the Dense Tensor
# Useful for using Base Array functions
Base.Array(T::DenseTensor) = reshape(data(store(T)),dims(inds(T)))
Base.Matrix(T::DenseTensor{<:Number,2}) = Array(T)
Base.Vector(T::DenseTensor{<:Number,1}) = Array(T)

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function Base.permutedims!(R::DenseTensor{<:Number,N},
                           T::DenseTensor{<:Number,N},
                           perm::NTuple{N,Int}) where {N}
  permutedims!(Array(R),Array(T),perm)
  return R
end

# Version that may overwrite the result or promote
# and return the result
function permutedims!!(R::DenseTensor{<:Number,N},
                       T::DenseTensor{<:Number,N},
                       perm::NTuple{N,Int},f=(r,t)->t) where {N}
  permutedims!(R,T,perm,f)
  return R
end

# TODO: move to tensor.jl?
function Base.permutedims(T::Tensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  Tp = similar(T,permute(inds(T),perm))
  permutedims!(Tp,T,perm)
  return Tp
end

# For use in custom permutedims!
using Base.Cartesian: @nexprs,
                      @ntuple,
                      @nloops

#
# A generalized permutedims!(P,B,perm) that also allows
# a function to be applied elementwise
# TODO: benchmark to make sure it is similar to Base.permutedims!
#
# Based off of the permutedims! implementation in Julia's base:
# https://github.com/JuliaLang/julia/blob/91151ab871c7e7d6689d1cfa793c12062d37d6b6/base/multidimensional.jl#L1355
#
@generated function Base.permutedims!(TP::DenseTensor{<:Number,N},
                                      T::DenseTensor{<:Number,N},
                                      perm,
                                      f::Function) where {N}
  quote
    Base.checkdims_perm(TP, T, perm)

    #calculates all the strides
    native_strides = Base.size_to_strides(1, size(T)...)
    strides_1 = 0
    @nexprs $N d->(strides_{d+1} = native_strides[perm[d]])

    #Creates offset, because indexing starts at 1
    offset = 1 - sum(@ntuple $N d->strides_{d+1})

    ind = 1
    @nexprs 1 d->(counts_{$N+1} = strides_{$N+1}) # a trick to set counts_($N+1)
    @nloops($N, i, TP,
            d->(counts_d = strides_d), # PRE
            d->(counts_{d+1} += strides_{d+1}), # POST
            begin # BODY
                sumc = sum(@ntuple $N d->counts_{d+1})
                @inbounds TP[ind] = f(TP[ind],T[sumc+offset])
                ind += 1
            end)

    return TP
  end
end

function outer(T1::DenseTensor,T2::DenseTensor)
  return Tensor(Dense(vec(Vector(T1)*transpose(Vector(T2)))),union(inds(T1),inds(T2)))
end
const ⊗ = outer

# TODO: move to tensor.jl?
function contract(T1::Tensor,labelsT1,
                  T2::Tensor,labelsT2)
  indsR,labelsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2)
  R = similar(promote_type(typeof(T1),typeof(T2)),indsR)
  contract!(R,labelsR,T1,labelsT1,T2,labelsT2)
  return R
end

function contract!(R::DenseTensor,labelsR,
                   T1::DenseTensor{<:Number,N1},labelsT1,
                   T2::DenseTensor{<:Number,N2},labelsT2) where {N1,N2}
  if N1==0
    # TODO: replace with an add! function?
    # What about doing `R .= T1[] .* PermutedDimsArray(T2,perm)`?
    perm = getperm(labelsR,labelsT2)
    permutedims!(R,T2,perm,(r,t2)->T1[]*t2)
  elseif N2==0
    perm = getperm(labelsR,labelsT1)
    permutedims!(R,T1,perm,(r,t1)->T2[]*t1)
  elseif isdisjoint(labelsT1,labelsT2)
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    Rp = T1⊗T2
    labelsRp = union(labelsT1,labelsT2)
    perm = getperm(labelsR,labelsRp)
    if !istrivial(perm)
      permutedims!(R,Rp,perm)
    else
      copyto!(R,Rp)
    end
  else
    _contract!(R,labelsR,T1,labelsT1,T2,labelsT2)
  end
  return R
end

# TODO: make sure this is doing type promotion correctly
# since we are calling BLAS (need to promote T1 and T2 to
# the same types)
function _contract!(R::DenseTensor,labelsR,
                    T1::DenseTensor,labelsT1,
                    T2::DenseTensor,labelsT2)
  props = ContractionProperties(labelsT1,labelsT2,labelsR)
  compute_contraction_properties!(props,T1,T2,R)
  _contract!(R,T1,T2,props)
  return R
end

function _contract!(C::DenseTensor{T},
                    A::DenseTensor{T},
                    B::DenseTensor{T},
                    props::ContractionProperties,
                    α::T=one(T),
                    β::T=zero(T)) where {T}
  tA = 'N'
  if props.permuteA
    aref = reshape(permutedims(A,props.PA),props.dmid,props.dleft)
    tA = 'T'
  else
    #A doesn't have to be permuted
    if Atrans(props)
      aref = reshape(A,props.dmid,props.dleft)
      tA = 'T'
    else
      aref = reshape(A,props.dleft,props.dmid)
    end
  end

  tB = 'N'
  if props.permuteB
    bref = reshape(permutedims(B,props.PB),props.dmid,props.dright)
  else
    if Btrans(props)
      bref = reshape(B,props.dright,props.dmid)
      tB = 'T'
    else
      bref = reshape(B,props.dmid,props.dright)
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    cref = reshape(copy(C),props.dleft,props.dright)
  else
    if Ctrans(props)
      cref = reshape(C,props.dleft,props.dright)
      if tA=='N' && tB=='N'
        (aref,bref) = (bref,aref)
        tA = tB = 'T'
      elseif tA=='T' && tB=='T'
        (aref,bref) = (bref,aref)
        tA = tB = 'N'
      end
    else
      cref = reshape(C,props.dleft,props.dright)
    end
  end

  #BLAS.gemm!(tA,tB,promote_type(T,Tα)(α),aref,bref,promote_type(T,Tβ)(β),cref)
  BLAS.gemm!(tA,tB,α,aref,bref,β,cref)

  if props.permuteC
    permutedims!(C,reshape(cref,props.newCrange...),props.PC)
  end
  return C
end

# Combine a bunch of tuples
# TODO: move this functionality to IndexSet, combine with unioninds?
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

"""
permute_reshape(T::Tensor,pos)

Takes a permutation that is split up into tuples. Index positions
within the tuples are combined.

For example:

permute_reshape(T,(3,2),1)

First T is permuted as `permutedims(3,2,1)`, then reshaped such
that the original indices 3 and 2 are combined.
"""
function permute_reshape(T::DenseTensor{ElT,NT,IndsT},pos::Vararg{<:Any,N}) where {ElT,NT,IndsT,N}
  perm = tuplejoin(pos...)

  length(perm)≠NT && error("Index positions must add up to order of Tensor ($N)")
  isperm(perm) || error("Index positions must be a permutation")

  dimsT = dims(T)
  indsT = inds(T)
  if !is_trivial_permutation(perm)
    T = permutedims(T,perm)
  end
  N==NT && return T
  newdims = MVector(ntuple(_->eltype(IndsT)(1),Val(N)))
  for i ∈ 1:N
    if length(pos[i])==1
      # No reshape needed, just use the
      # original index
      newdims[i] = indsT[pos[i][1]]
    else
      newdim_i = 1
      for p ∈ pos[i]
        newdim_i *= dimsT[p]
      end
      newdims[i] = eltype(IndsT)(newdim_i)
    end
  end
  newinds = similar_type(IndsT,Val(N))(Tuple(newdims))
  return reshape(T,newinds)
end

# svd of an order-n tensor according to positions Lpos
# and Rpos
function LinearAlgebra.svd(T::DenseTensor{<:Number,N,IndsT},
                           Lpos::NTuple{NL,Int},
                           Rpos::NTuple{NR,Int};
                           kwargs...) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  UM,S,VM = svd(M;kwargs...)
  u = ind(UM,2)
  v = ind(VM,2)
  
  Linds = similar_type(IndsT,Val(NL))(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Uinds = push(Linds,u)

  # TODO: do these positions need to be reversed?
  Rinds = similar_type(IndsT,Val(NR))(ntuple(i->inds(T)[Rpos[i]],Val(NR)))
  Vinds = push(Rinds,v)

  U = reshape(UM,Uinds)
  V = reshape(VM,Vinds)

  return U,S,V
end

# svd of an order-2 tensor
function LinearAlgebra.svd(T::DenseTensor{ElT,2,IndsT};
                           kwargs...) where {ElT,IndsT}
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  fastSVD::Bool = get(kwargs,:fastSVD,false)

  if fastSVD
    MU,MS,MV = svd(Matrix(T))
  else
    MU,MS,MV = recursiveSVD(Matrix(T))
  end
  conj!(MV)

  P = MS.^2
  truncate!(P;mindim=mindim,
              maxdim=maxdim,
              cutoff=cutoff,
              absoluteCutoff=absoluteCutoff,
              doRelCutoff=doRelCutoff)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    resize!(MS,dS)
    MV = MV[:,1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T,1),u))
  Sinds = IndsT((u,v))
  Vinds = IndsT((ind(T,2),v))
  U = Tensor(Dense{ElT}(vec(MU)),Uinds)
  S = Tensor(Diag{Vector{real(ElT)}}(MS),Sinds)
  V = Tensor(Dense{ElT}(vec(MV)),Vinds)
  return U,S,V
end

# eigendecomposition of an order-n tensor according to positions Lpos
# and Rpos
function eigenHermitian(T::DenseTensor{<:Number,N,IndsT},
                        Lpos::NTuple{NL,Int},
                        Rpos::NTuple{NR,Int};
                        kwargs...) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  UM,D = eigenHermitian(M;kwargs...)
  u = ind(UM,2)
  Linds = similar_type(IndsT,Val(NL))(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Uinds = push(Linds,u)
  U = reshape(UM,Uinds)
  return U,D
end

function eigenHermitian(T::DenseTensor{ElT,2,IndsT};
                        kwargs...) where {ElT,IndsT}
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  #tags::TagSet = get(kwargs,:lefttags,"Link,u")
  #lefttags::TagSet = get(kwargs,:lefttags,tags)
  #righttags::TagSet = get(kwargs,:righttags,prime(lefttags))

  DM,UM = eigen(Hermitian(Matrix(T)))

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; rev = true)
  DM = DM[p]
  UM = UM[:,p]

  truncate!(DM;maxdim=maxdim,
               cutoff=cutoff,
               absoluteCutoff=absoluteCutoff,
               doRelCutoff=doRelCutoff)
  dD = length(DM)
  if dD < size(UM,2)
    UM = UM[:,1:dD]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dD)
  v = eltype(IndsT)(dD)
  Uinds = IndsT((ind(T,1),u))
  Dinds = IndsT((u,v))
  U = Tensor(Dense{ElT}(vec(UM)),Uinds)
  D = Tensor(Diag{Vector{real(ElT)}}(DM),Dinds)
  return U,D
end

# qr decomposition of an order-n tensor according to positions Lpos
# and Rpos
function LinearAlgebra.qr(T::DenseTensor{<:Number,N,IndsT},
                          Lpos::NTuple{NL,Int},
                          Rpos::NTuple{NR,Int}) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  QM,RM = qr(M)
  q = ind(QM,2)
  r = ind(RM,1)
  Linds = similar_type(IndsT,Val(NL))(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Qinds = push(Linds,q)
  Q = reshape(QM,Qinds)
  Rinds = similar_type(IndsT,Val(NR))(ntuple(i->inds(T)[Rpos[i]],Val(NR)))
  Rinds = pushfirst(Rinds,r)
  R = reshape(RM,Rinds)
  return Q,R
end

function LinearAlgebra.qr(T::DenseTensor{ElT,2,IndsT};
                          kwargs...) where {ElT,IndsT}
  QM,RM = qr(Matrix(T))
  dim = size(QM,2) 
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  Q = Tensor(Dense{ElT}(vec(Matrix(QM))),Qinds)
  R = Tensor(Dense{ElT}(vec(Matrix(RM))),Rinds)
  return Q,R
end

# polar decomposition of an order-n tensor according to positions Lpos
# and Rpos
function polar(T::DenseTensor{<:Number,N,IndsT},
               Lpos::NTuple{NL,Int},
               Rpos::NTuple{NR,Int}) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  UM,PM = polar(M)

  Linds = similar_type(IndsT,Val(NL))(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Rinds = similar_type(IndsT,Val(NR))(ntuple(i->inds(T)[Rpos[i]],Val(NR)))

  # Use sim to create "similar" indices, in case
  # the indices have identifiers. If not this should
  # act as an identity operator
  Rinds_sim = sim(Rinds)

  Uinds = unioninds(Linds,Rinds_sim)
  Pinds = unioninds(Rinds_sim,Rinds)

  U = reshape(UM,Uinds)
  P = reshape(PM,Pinds)
  return U,P
end

function polar(T::DenseTensor{ElT,2,IndsT};
               kwargs...) where {ElT,IndsT}
  QM,RM = polar(Matrix(T))
  dim = size(QM,2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  Q = Tensor(Dense{ElT}(vec(Matrix(QM))),Qinds)
  R = Tensor(Dense{ElT}(vec(Matrix(RM))),Rinds)
  return Q,R
end

