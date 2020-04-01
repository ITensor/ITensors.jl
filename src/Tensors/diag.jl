export Diag,
       DiagTensor

# Diag can have either Vector storage, in which case
# it is a general Diag tensor, or scalar storage,
# in which case the diagonal has a uniform value
struct Diag{ElT,VecT} <: TensorStorage{ElT}
  data::VecT
  function Diag{ElT,VecT}(data) where {ElT,VecT<:AbstractVector{ElT}}
    return new{ElT,VecT}(data)
  end
  function Diag{ElT,ElT}(data) where {ElT}
    return new{ElT,ElT}(data)
  end
end

Diag(data::VecT) where {VecT<:AbstractVector{ElT}} where {ElT} = Diag{ElT,VecT}(data)

Diag(data::ElT) where {ElT<:Number} = Diag{ElT,ElT}(data)

function Diag{ElR}(data::AbstractVector{ElT}) where {ElR<:Number,ElT<:Number}
  ElT == ElR ? Diag(data) : Diag(ElR.(data))
end

Diag(::Type{ElT},
     n::Integer) where {ElT<:Number} = Diag(zeros(ElT,n))

Diag(x::ElT,
     n::Integer) where {ElT<:Number} = Diag(fill(x,n))

Base.copy(D::Diag) = Diag(copy(data(D)))

const NonuniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:AbstractVector}

const UniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:Number}

Base.getindex(D::UniformDiag,i::Int) = data(D)

Base.setindex!(D::UniformDiag,val,i::Int) = error("Cannot set elements of a uniform Diag storage")

Base.complex(::Type{Diag{ElT,Vector{ElT}}}) where {ElT} = Diag{complex(ElT),Vector{complex(ElT)}}
Base.complex(::Type{Diag{ElT,ElT}}) where {ElT} = Diag{complex(ElT),complex(ElT)}

# Deal with uniform Diag conversion
Base.convert(::Type{<:Diag{ElT,VecT}},D::Diag) where {ElT,VecT} = Diag(convert(VecT,data(D)))

# TODO: write in terms of ::Int, not inds
Base.similar(D::NonuniformDiag) = Diag(similar(data(D)))
#Base.similar(D::NonuniformDiag,inds) = Diag(similar(data(D),minimum(dims(inds))))
#function Base.similar(D::Type{<:NonuniformDiag{ElT,VecT}},inds) where {ElT,VecT}
#  return Diag(similar(VecT,diaglength(inds)))
#end

Base.similar(D::UniformDiag) = Diag(zero(T))
Base.similar(D::UniformDiag,inds) = similar(D)
Base.similar(::Type{<:UniformDiag{ElT}},inds) where {ElT} = Diag(zero(ElT))

Base.similar(D::Diag,n::Int) = Diag(similar(data(D),n))

Base.similar(D::Diag,::Type{ElR},n::Int) where {ElR} = Diag(similar(data(D),ElR,n))

# TODO: make this work for other storage besides Vector
Base.zeros(::Type{<:NonuniformDiag{ElT}},dim::Int64) where {ElT} = Diag(zeros(ElT,dim))
Base.zeros(::Type{<:UniformDiag{ElT}},dim::Int64) where {ElT} = Diag(zero(ElT))

Base.:*(D::Diag,x::Number) = Diag(x*data(D))
Base.:*(x::Number,D::Diag) = D*x

#
# Type promotions involving Diag
# Useful for knowing how conversions should work when adding and contracting
#

function Base.promote_rule(::Type{<:UniformDiag{ElT1}},
                           ::Type{<:UniformDiag{ElT2}}) where {ElT1,ElT2}
  ElR = promote_type(ElT1,ElT2)
  return Diag{ElR,ElR}
end

function Base.promote_rule(::Type{<:NonuniformDiag{ElT1,VecT1}},
                           ::Type{<:NonuniformDiag{ElT2,VecT2}}) where {ElT1,VecT1<:AbstractVector,
                                                                        ElT2,VecT2<:AbstractVector}
  ElR = promote_type(ElT1,ElT2)
  VecR = promote_type(VecT1,VecT2)
  return Diag{ElR,VecR}
end

# This is an internal definition, is there a more general way?
#Base.promote_type(::Type{Vector{ElT1}},
#                  ::Type{ElT2}) where {ElT1<:Number,
#                                       ElT2<:Number} = Vector{promote_type(ElT1,ElT2)}
#
#Base.promote_type(::Type{ElT1},
#                  ::Type{Vector{ElT2}}) where {ElT1<:Number,
#                                               ElT2<:Number} = promote_type(Vector{ElT2},ElT1)

# TODO: how do we make this work more generally for T2<:AbstractVector{S2}?
# Make a similar_type(AbstractVector{S2},T1) -> AbstractVector{T1} function?
function Base.promote_rule(::Type{<:UniformDiag{ElT1,VecT1}},
                           ::Type{<:NonuniformDiag{ElT2,Vector{ElT2}}}) where {ElT1,VecT1<:Number,
                                                                               ElT2}
  ElR = promote_type(ElT1,ElT2)
  VecR = Vector{ElR}
  return Diag{ElR,VecR}
end

function Base.promote_rule(::Type{DenseT1},
                           ::Type{<:NonuniformDiag{ElT2,VecT2}}) where {DenseT1<:Dense,
                                                                        ElT2,VecT2<:AbstractVector}
  return promote_type(DenseT1,Dense{ElT2,VecT2})
end

function Base.promote_rule(::Type{DenseT1},
                           ::Type{<:UniformDiag{ElT2,VecT2}}) where {DenseT1<:Dense,
                                                                     ElT2,VecT2<:Number}
  return promote_type(DenseT1,ElT2)
end

# Convert a Diag storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiag{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}
dense(::Type{<:UniformDiag{ElT}}) where {ElT} = Dense{ElT,Vector{ElT}}

const DiagTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Diag}
const NonuniformDiagTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where 
                                               {StoreT<:NonuniformDiag}
const UniformDiagTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where 
                                               {StoreT<:UniformDiag}

Base.IndexStyle(::Type{<:DiagTensor}) = IndexCartesian()

# TODO: this needs to be better (promote element type, check order compatibility,
# etc.
function Base.convert(::Type{<:DenseTensor{ElT,N}}, T::DiagTensor{ElT,N}) where {ElT<:Number,N}
  return dense(T)
end

# These are rules for determining the output of a pairwise contraction of Tensors
# (given the indices of the output tensors)
function contraction_output_type(TensorT1::Type{<:DiagTensor},
                                 TensorT2::Type{<:DenseTensor},
                                 IndsR::Type)
  return similar_type(promote_type(TensorT1,TensorT2),IndsR)
end
contraction_output_type(TensorT1::Type{<:DenseTensor},
                        TensorT2::Type{<:DiagTensor},
                        IndsR::Type) = contraction_output_type(TensorT2,TensorT1,IndsR)

# This performs the logic that DiagTensor*DiagTensor -> DiagTensor if it is not an outer
# product but -> DenseTensor if it is
# TODO: if the tensors are both order 2 (or less), or if there is an Index replacement,
# then they remain diagonal. Should we limit DiagTensor*DiagTensor to cases that
# result in a DiagTensor, for efficiency and type stability? What about a general
# SparseTensor result?
function contraction_output_type(TensorT1::Type{<:DiagTensor{<:Number,N1}},
                                 TensorT2::Type{<:DiagTensor{<:Number,N2}},
                                 IndsR::Type) where {N1,N2}
  if ValLength(IndsR)===Val{N1+N2}
    # Turn into is_outer(inds1,inds2,indsR) function?
    # How does type inference work with arithmatic of compile time values?
    return similar_type(dense(promote_type(TensorT1,TensorT2)),IndsR)
  end
  return similar_type(promote_type(TensorT1,TensorT2),IndsR)
end

# The output must be initialized as zero since it is sparse, cannot be undefined
contraction_output(T1::DiagTensor,T2::Tensor,indsR) = zero_contraction_output(T1,T2,indsR)
contraction_output(T1::Tensor,T2::DiagTensor,indsR) = contraction_output(T2,T1,indsR)

function contraction_output(T1::DiagTensor,
                            T2::DiagTensor,
                            indsR)
  return zero_contraction_output(T1,T2,indsR)
end

function array(T::DiagTensor{ElT,N}) where {ElT,N}
  return array(dense(T))
end
matrix(T::DiagTensor{<:Number,2}) = array(T)
vector(T::DiagTensor{<:Number,1}) = array(T)

function Base.Array{ElT,N}(T::DiagTensor{ElT,N}) where {ElT,N}
  return array(T)
end

function Base.Array(T::DiagTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

# Needed to get slice of DiagTensor like T[1:3,1:3]
function Base.similar(T::DiagTensor{<:Number,N},
                      ::Type{ElR},
                      inds::Dims{N}) where {ElR<:Number,N}
  return Tensor(similar(store(T),ElR,minimum(inds)),inds)
end

"""
getdiagindex(T::DiagTensor,i::Int)

Get the ith value along the diagonal of the tensor.
"""
getdiagindex(T::DiagTensor{<:Number},ind::Int) = store(T)[ind]

"""
setdiagindex!(T::DiagTensor,i::Int)

Set the ith value along the diagonal of the tensor.
"""
setdiagindex!(T::DiagTensor{<:Number},val,ind::Int) = (store(T)[ind] = val)

"""
setdiag(T::UniformDiagTensor,val)

Set the entire diagonal of a uniform DiagTensor.
"""
setdiag(T::UniformDiagTensor,val) = Tensor(Diag(val),inds(T))

Base.@propagate_inbounds function Base.getindex(T::DiagTensor{ElT,N},
                                                inds::Vararg{Int,N}) where {ElT,N}
  if all(==(inds[1]),inds)
    return getdiagindex(T,inds[1])
  else
    return zero(eltype(ElT))
  end
end
Base.@propagate_inbounds Base.getindex(T::DiagTensor{<:Number,1},ind::Int) = store(T)[ind]
Base.@propagate_inbounds Base.getindex(T::DiagTensor{<:Number,0}) = store(T)[1]

# Set diagonal elements
# Throw error for off-diagonal
Base.@propagate_inbounds function Base.setindex!(T::DiagTensor{<:Number,N},
                                                 val,inds::Vararg{Int,N}) where {N}
  all(==(inds[1]),inds) || error("Cannot set off-diagonal element of Diag storage")
  setdiagindex!(T,val,inds[1])
  return T
end
Base.@propagate_inbounds Base.setindex!(T::DiagTensor{<:Number,1},val,ind::Int) = ( store(T)[ind] = val )
Base.@propagate_inbounds Base.setindex!(T::DiagTensor{<:Number,0},val) = ( store(T)[1] = val )

function Base.setindex!(T::UniformDiagTensor{<:Number,N},val,inds::Vararg{Int,N}) where {N}
  error("Cannot set elements of a uniform Diag storage")
end

# TODO: make a fill!! that works for uniform and non-uniform
#Base.fill!(T::DiagTensor,v) = fill!(store(T),v)

function dense(::Type{<:Tensor{ElT,N,StoreT,IndsT}}) where {ElT,N,
                                                            StoreT<:Diag,IndsT}
  return Tensor{ElT,N,dense(StoreT),IndsT}
end

# convert to Dense
function dense(T::TensorT) where {TensorT<:DiagTensor}
  R = zeros(dense(TensorT),inds(T))
  for i = 1:diaglength(T)
    setdiagindex!(R,getdiagindex(T,i),i)
  end
  return R
end

function outer!(R::DenseTensor{<:Number,NR},
                T1::DiagTensor{<:Number,N1},
                T2::DiagTensor{<:Number,N2}) where {NR,N1,N2}
  for i1 = 1:diaglength(T1), i2 = 1:diaglength(T2)
    indsR = CartesianIndex{NR}(ntuple(r -> r ≤ N1 ? i1 : i2, Val(NR)))
    R[indsR] = getdiagindex(T1,i1)*getdiagindex(T2,i2)
  end
  return R
end

# TODO: write an optimized version of this?
function outer!(R::DenseTensor{ElR},
                T1::DenseTensor,
                T2::DiagTensor) where {ElR}
  R .= zero(ElR)
  outer!(R,T1,dense(T2))
  return R
end

function outer!(R::DenseTensor{ElR},
                T1::DiagTensor,
                T2::DenseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R,dense(T1),T2)
  return R
end

# Right an in-place version
function outer(T1::DiagTensor{ElT1,N1},
               T2::DiagTensor{ElT2,N2}) where {ElT1,ElT2,N1,N2}
  indsR = unioninds(inds(T1),inds(T2))
  R = Tensor(Dense(zeros(promote_type(ElT1,ElT2),dim(indsR))),indsR)
  outer!(R,T1,T2)
  return R
end

function Base.permutedims!(R::DiagTensor{<:Number,N},
                           T::DiagTensor{<:Number,N},
                           perm::NTuple{N,Int},f::Function=(r,t)->t) where {N}
  # TODO: check that inds(R)==permute(inds(T),perm)?
  for i=1:diaglength(R)
    @inbounds setdiagindex!(R,f(getdiagindex(R,i),getdiagindex(T,i)),i)
  end
  return R
end

function Base.permutedims(T::DiagTensor{<:Number,N},
                          perm::NTuple{N,Int},f::Function=identity) where {N}
  R = similar(T,permute(inds(T),perm))
  permutedims!(R,T,perm,f)
  return R
end

function Base.permutedims(T::UniformDiagTensor{ElT,N},
                          perm::NTuple{N,Int},
                          f::Function=identity) where {ElR,ElT,N}
  R = Tensor(Diag(f(getdiagindex(T,1))),permute(inds(T),perm))
  return R
end

# Version that may overwrite in-place or may return the result
function permutedims!!(R::NonuniformDiagTensor{<:Number,N},
                       T::NonuniformDiagTensor{<:Number,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {N}
  permutedims!(R,T,perm,f)
  return R
end

function permutedims!!(R::UniformDiagTensor{ElR,N},
                       T::UniformDiagTensor{ElT,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {ElR,ElT,N}
  R = Tensor(Diag(f(getdiagindex(R,1),getdiagindex(T,1))),inds(R))
  return R
end

function Base.permutedims!(R::DenseTensor{ElR,N},
                           T::DiagTensor{ElT,N},
                           perm::NTuple{N,Int},
                           f::Function = (r,t)->t) where {ElR,ElT,N}
  for i = 1:diaglength(T)
    @inbounds setdiagindex!(R,f(getdiagindex(R,i),getdiagindex(T,i)),i)
  end
  return R
end

function permutedims!!(R::DenseTensor{ElR,N},
                       T::DiagTensor{ElT,N},
                       perm::NTuple{N,Int},f::Function=(r,t)->t) where {ElR,ElT,N}
  permutedims!(R,T,perm,f)
  return R
end

function _contract!!(R::UniformDiagTensor{ElR,NR},labelsR,
                     T1::UniformDiagTensor{<:Number,N1},labelsT1,
                     T2::UniformDiagTensor{<:Number,N2},labelsT2) where {ElR,NR,N1,N2}
  if NR==0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    R = setdiag(R,diaglength(T1)*getdiagindex(T1,1)*getdiagindex(T2,1))
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    R = setdiag(R,getdiagindex(T1,1)*getdiagindex(T2,1))
  end
  return R
end

function contract!(R::DiagTensor{ElR,NR},labelsR,
                   T1::DiagTensor{<:Number,N1},labelsT1,
                   T2::DiagTensor{<:Number,N2},labelsT2) where {ElR,NR,N1,N2}
  if NR==0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    Rdiag = zero(ElR)
    for i = 1:diaglength(T1)
      Rdiag += getdiagindex(T1,i)*getdiagindex(T2,i)
    end
    setdiagindex!(R,Rdiag,1)
  else
    min_dim = min(diaglength(T1),diaglength(T2))
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    for i = 1:min_dim
      setdiagindex!(R,getdiagindex(T1,i)*getdiagindex(T2,i),i)
    end
  end
  return R
end

function contract!(C::DenseTensor{ElC,NC},Clabels,
                   A::DiagTensor{ElA,NA},Alabels,
                   B::DenseTensor{ElB,NB},Blabels) where {ElA,NA,
                                                          ElB,NB,
                                                          ElC,NC}
  if all(i -> i < 0, Blabels)
    # If all of B is contracted
    # TODO: can also check NC+NB==NA
    min_dim = minimum(dims(B))
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      for i = 1:min_dim
        setdiagindex!(C,getdiagindex(C,1)+getdiagindex(A,i)*getdiagindex(B,i),1)
      end
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i = 1:min_dim
        setdiagindex!(C,getdiagindex(A,i)*getdiagindex(B,i),i)
      end
    end
  else
    astarts = zeros(Int,length(Alabels))
    bstart = 0
    cstart = 0
    b_cstride = 0
    nbu = 0
    for ib = 1:length(Blabels)
      ia = findfirst(==(Blabels[ib]),Alabels)
      if !isnothing(ia)
        b_cstride += stride(B,ib)
        bstart += astarts[ia]*stride(B,ib)
      else
        nbu += 1
      end
    end

    c_cstride = 0
    for ic = 1:length(Clabels)
      ia = findfirst(==(Clabels[ic]),Alabels)
      if !isnothing(ia)
        c_cstride += stride(C,ic)
        cstart += astarts[ia]*stride(C,ic)
      end
    end

    # strides of the uncontracted dimensions of
    # B
    bustride = zeros(Int,nbu)
    custride = zeros(Int,nbu)
    # size of the uncontracted dimensions of
    # B, to be used in CartesianIndices
    busize = zeros(Int,nbu)
    n = 1
    for ib = 1:length(Blabels)
      if Blabels[ib] > 0
        bustride[n] = stride(B,ib)
        busize[n] = size(B,ib)
        ic = findfirst(==(Blabels[ib]),Clabels)
        custride[n] = stride(C,ic)
        n += 1
      end
    end

    boffset_orig = 1-sum(strides(B))
    coffset_orig = 1-sum(strides(C))
    cartesian_inds = CartesianIndices(Tuple(busize))
    for inds in cartesian_inds
      boffset = boffset_orig
      coffset = coffset_orig
      for i in 1:nbu
        ii = inds[i]
        boffset += ii*bustride[i]
        coffset += ii*custride[i]
      end
      for j in 1:diaglength(A)
        C[cstart+j*c_cstride+coffset] += getdiagindex(A,j)*
                                         B[bstart+j*b_cstride+boffset]
      end
    end
  end
end
contract!(C::DenseTensor,Clabels,
          A::DenseTensor,Alabels,
          B::DiagTensor,Blabels) = contract!(C,Clabels,
                                             B,Blabels,
                                             A,Alabels)

