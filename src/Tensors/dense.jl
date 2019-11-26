export Dense,
       DenseTensor,
       outer,
       ⊗

#
# Dense storage
#

struct Dense{ElT,VecT<:AbstractVector} <: TensorStorage{ElT}
  data::VecT
  Dense(data::VecT) where {VecT<:AbstractVector{ElT}} where {ElT} = new{ElT,VecT}(data)
end

Dense{ElT}(dim::Integer) where {ElT} = Dense(zeros(ElT,dim))

Dense(dim::Integer) where {ElT} = Dense{Float64}(dim)

Dense{ElT}(::UndefInitializer,dim::Integer) where {ElT} = Dense(Vector{ElT}(undef,dim))

Dense(::UndefInitializer,dim::Integer) = Dense{Float64}(undef,dim)

function Dense{ElR}(data::VecT) where {ElR,VecT<:AbstractVector{ElT}} where {ElT}
  ElT == ElR ? Dense(data) : Dense(ElR.(data))
end
Dense{ElT}() where {ElT} = Dense(ElT[])

# Convenient functions for Dense storage type
Base.@propagate_inbounds Base.getindex(D::Dense,i::Integer) = data(D)[i]
Base.@propagate_inbounds Base.setindex!(D::Dense,v,i::Integer) = (data(D)[i] = v)

Base.similar(D::Dense) = Dense(similar(data(D)))

Base.similar(D::Dense,length::Int) = Dense(similar(data(D),length))
Base.similar(::Type{<:Dense{ElT,VecT}},length::Int) where {ElT,VecT} = Dense(similar(VecT,length))

Base.similar(D::Dense,::Type{T}) where {T<:Number} = Dense(similar(data(D),T))
Base.copy(D::Dense) = Dense(copy(data(D)))
Base.copyto!(D1::Dense,D2::Dense) = copyto!(data(D1),data(D2))

Base.fill!(D::Dense,v) = fill!(data(D),v)

# TODO: should this do something different for SubArray?
Base.zeros(::Type{<:Dense{ElT}},dim::Int) where {ElT} = Dense{ElT}(dim)

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::Dense) = Dense(complex(data(D)))

Base.eltype(::Dense{ElT}) where {ElT} = ElT
# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::Dense{Nothing}) = Nothing
Base.eltype(::Type{<:Dense{ElT}}) where {ElT} = ElT

function Base.promote_rule(::Type{<:Dense{ElT1,VecT1}},
                           ::Type{<:Dense{ElT2,VecT2}}) where {ElT1,VecT1,
                                                               ElT2,VecT2}
  ElR = promote_type(ElT1,ElT2)
  VecR = promote_type(VecT1,VecT2)
  return Dense{ElR,VecR}
end

# This is to get around the issue in Julia that:
# promote_type(Vector{ComplexF32},Vector{Float64}) == Vector{T} where T
function Base.promote_rule(::Type{<:Dense{ElT1,Vector{ElT1}}},
                           ::Type{<:Dense{ElT2,Vector{ElT2}}}) where {ElT1,ElT2}
  ElR = promote_type(ElT1,ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

# This is for type promotion for Scalar*Dense
function Base.promote_rule(::Type{<:Dense{ElT1,Vector{ElT1}}},
                           ::Type{ElT2}) where {ElT1,
                                                ElT2<:Number}
  ElR = promote_type(ElT1,ElT2)
  VecR = Vector{ElR}
  return Dense{ElR,VecR}
end

Base.convert(::Type{<:Dense{ElR,VecR}},
             D::Dense) where {ElR,VecR} = Dense(convert(VecR,data(D)))

Base.:*(D::Dense,x::Number) = Dense(x*data(D))
Base.:*(x::Number,D::Dense) = D*x

#
# DenseTensor (Tensor using Dense storage)
#

const DenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Dense}

DenseTensor(inds::Dims{N}) where {N} = Tensor(Dense(dim(inds)),inds)

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:DenseTensor}) = IndexLinear()

# TODO: make a version accepting I... with some mixture of ranges
# Note that CartesianIndices((1:2,3)) creates CartesianIndex(1,3):CartesianIndex(2,3)
Base.@propagate_inbounds function Base.getindex(D::DenseTensor{ElT,N},
                                                I::CartesianIndices{N}) where {ElT,N}
  storeR = Dense(vec(@view array(D)[I]))
  indsR = Tuple(I[end]-I[1]+CartesianIndex(ntuple(_->1,Val(N))))
  return Tensor(storeR,indsR)
end

Base.@propagate_inbounds Base.getindex(T::DenseTensor,i::Int) = store(T)[i]

Base.@propagate_inbounds Base.setindex!(T::DenseTensor,v,i::Int) = (store(T)[i] = v)

Base.fill!(T::DenseTensor,v) = fill!(store(T),v)

# How does Julia map from IndexCartesian to IndexLinear?
#Base.getindex(T::DenseTensor{<:Number,N},
#              i::Vararg{Int,N}) where {N} = 
#store(T)[sum(i.*strides(T))+1-sum(strides(T))]
#Base.setindex!(T::DenseTensor{<:Number,N},
#               v,i::Vararg{Int,N}) where {N} = 
#(store(T)[sum(i.*strides(T))+1-sum(strides(T))] = v)

# Get the specified value on the diagonal
function getdiag(T::DenseTensor{<:Number,N},ind::Int) where {N}
  return T[CartesianIndex(ntuple(_->ind,Val(N)))]
end

# Set the specified value on the diagonal
function setdiag!(T::DenseTensor{<:Number,N},val,ind::Int) where {N}
  T[CartesianIndex(ntuple(_->ind,Val(N)))] = val
end

# This is for BLAS/LAPACK
Base.strides(T::DenseTensor) = strides(inds(T))

# Needed for passing Tensor{T,2} to BLAS/LAPACK
function Base.unsafe_convert(::Type{Ptr{ElT}},
                             T::DenseTensor{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT},data(store(T)))
end

# Convert a Dense Tensor to a Tensor with the specified storage
function Base.convert(::Type{<:Tensor{<:Any,<:Any,StoreR}},
                      T::DenseTensor) where {StoreR}
  return Tensor(convert(StoreR,store(T)),inds(T))
end

# Reshape a DenseTensor using the specified dimensions
# This returns a view into the same Tensor data
function Base.reshape(T::DenseTensor,dims)
  dim(T)==dim(dims) || error("Total new dimension must be the same as the old dimension")
  return Tensor(store(T),dims)
end
# This version fixes method ambiguity with AbstractArray reshape
function Base.reshape(T::DenseTensor,dims::Dims)
  dim(T)==dim(dims) || error("Total new dimension must be the same as the old dimension")
  return Tensor(store(T),dims)
end
function Base.reshape(T::DenseTensor,dims::Int...)
  return Tensor(store(T),tuple(dims...))
end

# Create an Array that is a view of the Dense Tensor
# Useful for using Base Array functions
array(T::DenseTensor) = reshape(data(store(T)),dims(inds(T)))
matrix(T::DenseTensor{<:Number,2}) = array(T)
vector(T::DenseTensor{<:Number,1}) = array(T)

# TODO: call permutedims!(R,T,perm,(r,t)->t)?
function Base.permutedims!(R::DenseTensor{<:Number,N},
                           T::DenseTensor{<:Number,N},
                           perm::NTuple{N,Int}) where {N}
  permutedims!(array(R),array(T),perm)
  return R
end

function scale!(T::DenseTensor,
                α::Number)
  A = array(T)
  # This is faster than A .*= α
  rmul!(A,α)
  return T
end

# Version that may overwrite the result or promote
# and return the result
# TODO: move to tensor.jl?
function permutedims!!(R::Tensor,
                       T::Tensor,
                       perm::NTuple{N,Int},
                       f=(r,t)->t) where {N}
  if !is_trivial_permutation(perm)
    permutedims!(R,T,perm,f)
  else
    RA = array(R)
    TA = array(T)
    RA .= f.(RA,TA)
  end
  return R
end

# TODO: move to tensor.jl?
function Base.permutedims(T::Tensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  Tp = similar(T,permute(inds(T),perm))
  Tp = permutedims!!(Tp,T,perm)
  return Tp
end

# TODO: move to tensor.jl?
function Base.:*(x::Number,
                 T::Tensor)
  return Tensor(x*store(T),inds(T))
end
Base.:*(T::Tensor, x::Number) = x*T

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
@generated function Base.permutedims!(TTP::DenseTensor{<:Number,N},
                                      TT::DenseTensor{<:Number,N},
                                      perm,
                                      f::Function) where {N}
  quote
    TP = array(TTP)
    T = array(TT)
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

    return TTP
  end
end
function Base.permutedims!(TP::DenseTensor{<:Number,0},
                           T::DenseTensor{<:Number,0},
                           perm,
                           f::Function)
  TP[] = f(TP[],T[])
  return TP
end

function outer!(R::DenseTensor,
                T1::DenseTensor,
                T2::DenseTensor)
  v1 = vec(T1)
  v2 = vec(T2)
  RM = reshape(R,dim(v1),dim(v2))
  RM .= v1 .* transpose(v2)
  return R
end

function outer!!(R::Tensor,
                 T1::Tensor,
                 T2::Tensor)
  outer!(R,T1,T2)
  return R
end

# TODO: call outer!!, make this generic
function outer(T1::DenseTensor{ElT1},
               T2::DenseTensor{ElT2}) where {ElT1,ElT2}
  array_outer = vec(array(T1))*transpose(vec(array(T2)))
  inds_outer = unioninds(inds(T1),inds(T2))
  return Tensor(Dense{promote_type(ElT1,ElT2)}(vec(array_outer)),inds_outer)
end
const ⊗ = outer

# TODO: move to tensor.jl?
function contraction_output_type(::Type{TensorT1},
                                 ::Type{TensorT2},
                                 ::Type{IndsR}) where {TensorT1<:Tensor,
                                                       TensorT2<:Tensor,
                                                       IndsR}
  return similar_type(promote_type(TensorT1,TensorT2),IndsR)
end

function contraction_output(::TensorT1,
                            ::TensorT2,
                            indsR::IndsR) where {TensorT1<:DenseTensor,
                                                 TensorT2<:DenseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  return similar(TensorR,indsR)
end

# TODO: move to tensor.jl?
function contract(T1::Tensor{<:Any,N1},
                  labelsT1,
                  T2::Tensor{<:Any,N2},
                  labelsT2,
                  labelsR = contract_labels(labelsT1,labelsT2)) where {N1,N2}
  # TODO: put the contract_inds logic into contraction_output,
  # call like R = contraction_ouput(T1,labelsT1,T2,labelsT2)
  indsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2,labelsR)
  R = contraction_output(T1,T2,indsR)
  # contract!! version here since the output R may not
  # be mutable (like UniformDiag)
  R = contract!!(R,labelsR,T1,labelsT1,T2,labelsT2)
  return R
end

# Move to tensor.jl? Is this generic for all storage types?
function contract!!(R::Tensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::Tensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::Tensor{<:Number,N2},
                    labelsT2::NTuple{N2}) where {NR,N1,N2}
  if N1==0
    # TODO: replace with an add! function?
    # What about doing `R .= T1[] .* PermutedDimsArray(T2,perm)`?
    perm = getperm(labelsR,labelsT2)
    R = permutedims!!(R,T2,perm,(r,t2)->T1[]*t2)
  elseif N2==0
    perm = getperm(labelsR,labelsT1)
    R = permutedims!!(R,T1,perm,(r,t1)->T2[]*t1)
  elseif N1+N2==NR
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    # TODO: implement the in-place version directly
    R = outer!!(R,T1,T2)
    labelsRp = tuplecat(labelsT1,labelsT2)
    perm = getperm(labelsR,labelsRp)
    if !is_trivial_permutation(perm)
      R = permutedims!!(R,copy(R),perm)
    end
  else
    R = _contract!!(R,labelsR,T1,labelsT1,T2,labelsT2)
  end
  return R
end

# TODO: move to tensor.jl?
Base.copyto!(R::Tensor,T::Tensor) = copyto!(store(R),store(T))

# Move to tensor.jl? Overload this function
# for immutable storage types
function _contract!!(R::Tensor,labelsR,
                     T1::Tensor,labelsT1,
                     T2::Tensor,labelsT2)
  _contract!(R,labelsR,T1,labelsT1,T2,labelsT2)
  return R
end

# TODO: make sure this is doing type promotion correctly
# since we are calling BLAS (need to promote T1 and T2 to
# the same types)
function _contract!(R::DenseTensor,
                    labelsR,
                    T1::Tensor{<:Number,<:Any,StoreT1},
                    labelsT1,
                    T2::Tensor{<:Number,<:Any,StoreT2},
                    labelsT2) where {StoreT1<:Dense,StoreT2<:Dense}
  props = ContractionProperties(labelsT1,labelsT2,labelsR)
  compute_contraction_properties!(props,T1,T2,R)

  # We do type promotion here for BLAS (to ensure
  # we contract DenseComplex*DenseComplex)
  if StoreT1 !== StoreT2
    T1,T2 = promote(T1,T2)
  end

  _contract!(R,T1,T2,props)
  return R
end

function _contract!(CT::DenseTensor{El,NC},
                    AT::DenseTensor{El,NA},
                    BT::DenseTensor{El,NB},
                    props::ContractionProperties) where {El,NC,NA,NB}
  C = array(CT)
  A = array(AT)
  B = array(BT)

  tA = 'N'
  if props.permuteA
    AM = reshape(permutedims(A,NTuple{NA,Int}(props.PA)),props.dmid,props.dleft)
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
    BM = reshape(permutedims(B,NTuple{NB,Int}(props.PB)),props.dmid,props.dright)
  else
    if Btrans(props)
      BM = reshape(B,props.dright,props.dmid)
      tB = 'T'
    else
      BM = reshape(B,props.dmid,props.dright)
    end
  end

  # TODO: this logic may be wrong
  if props.permuteC
    # Need to copy here since we will be permuting
    # into C later
    CM = reshape(copy(C),props.dleft,props.dright)
  else
    if Ctrans(props)
      CM = reshape(C,props.dleft,props.dright)
      if tA=='N' && tB=='N'
        (AM,BM) = (BM,AM)
        tA = tB = 'T'
      elseif tA=='T' && tB=='T'
        (AM,BM) = (BM,AM)
        tA = tB = 'N'
      end
    else
      CM = reshape(C,props.dleft,props.dright)
    end
  end

  #BLAS.gemm!(tA,tB,promote_type(T,Tα)(α),AM,BM,promote_type(T,Tβ)(β),CM)

  # TODO: make sure this is fast with Tensor{ElT,2}, or
  # convert AM and BM to Matrix
  BLAS.gemm!(tA,tB,one(El),
             AM,BM,
             zero(El),CM)

  if props.permuteC
    permutedims!(C,reshape(CM,props.newCrange...),
                 NTuple{NC,Int}(props.PC))
  end
  return C
end

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
  perm = tuplecat(pos...)

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
  newinds = similar_type(IndsT,Val{N})(Tuple(newdims))
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
  
  Linds = similar_type(IndsT,Val{NL})(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Uinds = push(Linds,u)

  # TODO: do these positions need to be reversed?
  Rinds = similar_type(IndsT,Val{NR})(ntuple(i->inds(T)[Rpos[i]],Val(NR)))
  Vinds = push(Rinds,v)

  U = reshape(UM,Uinds)
  V = reshape(VM,Vinds)

  return U,S,V
end

# eigendecomposition of an order-n tensor according to 
# positions Lpos and Rpos
function eigenHermitian(T::DenseTensor{<:Number,N,IndsT},
                        Lpos::NTuple{NL,Int},
                        Rpos::NTuple{NR,Int};
                        kwargs...) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  UM,D = eigenHermitian(M;kwargs...)
  u = ind(UM,2)
  Linds = similar_type(IndsT,Val{NL})(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Uinds = push(Linds,u)
  U = reshape(UM,Uinds)
  return U,D
end

# qr decomposition of an order-n tensor according to 
# positions Lpos and Rpos
function LinearAlgebra.qr(T::DenseTensor{<:Number,N,IndsT},
                          Lpos::NTuple{NL,Int},
                          Rpos::NTuple{NR,Int}) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  QM,RM = qr(M)
  q = ind(QM,2)
  r = ind(RM,1)
  # TODO: simplify this by permuting inds(T) by (Lpos,Rpos)
  # then grab Linds,Rinds
  Linds = similar_type(IndsT,Val{NL})(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Qinds = push(Linds,r)
  Q = reshape(QM,Qinds)
  Rinds = similar_type(IndsT,Val{NR})(ntuple(i->inds(T)[Rpos[i]],Val(NR)))
  Rinds = pushfirst(Rinds,r)
  R = reshape(RM,Rinds)
  return Q,R
end

# polar decomposition of an order-n tensor according to positions Lpos
# and Rpos
function polar(T::DenseTensor{<:Number,N,IndsT},
               Lpos::NTuple{NL,Int},
               Rpos::NTuple{NR,Int}) where {N,IndsT,NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  UM,PM = polar(M)

  # TODO: turn these into functions
  Linds = similar_type(IndsT,Val{NL})(ntuple(i->inds(T)[Lpos[i]],Val(NL)))
  Rinds = similar_type(IndsT,Val{NR})(ntuple(i->inds(T)[Rpos[i]],Val(NR)))

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

function LinearAlgebra.exp(T::DenseTensor{ElT,N},
                           Lpos::NTuple{NL,Int},
                           Rpos::NTuple{NR,Int};
                           ishermitian::Bool=false) where {ElT,N,
                                                           NL,NR}
  M = permute_reshape(T,Lpos,Rpos)
  indsTp = permute(inds(T),(Lpos...,Rpos...))
  if ishermitian
    expM = exp(Hermitian(matrix(M)))
    return Tensor(Dense{ElT}(vec(expM)),indsTp)
  else
    expM = exp(M)
    return reshape(expM,indsTp)
  end
end

