export DiagBlockSparse,
       DiagBlockSparseTensor

# DiagBlockSparse can have either Vector storage, in which case
# it is a general DiagBlockSparse tensor, or scalar storage,
# in which case the diagonal has a uniform value
struct DiagBlockSparse{ElT,VecT,N} <: TensorStorage{ElT}
  data::VecT
  diagblockoffsets::BlockOffsets{N}  # Block number-offset pairs

  # Nonuniform case
  DiagBlockSparse(data::VecT,blockoffsets::BlockOffsets{N}) where {VecT<:AbstractVector{ElT},N} where {ElT} = new{ElT,VecT,N}(data,blockoffsets)

  # Uniform case
  DiagBlockSparse(data::VecT,blockoffsets::BlockOffsets{N}) where {VecT<:Number,N} = new{VecT,VecT,N}(data,blockoffsets)
end

function DiagBlockSparse(::Type{ElT},
                         boffs::BlockOffsets,
                         diaglength::Integer) where {ElT<:Number}
  return DiagBlockSparse(zeros(ElT,diaglength),boffs)
end

DiagBlockSparse(boffs::BlockOffsets,diaglength::Integer) = DiagBlockSparse(Float64,boffs,diaglength)

function DiagBlockSparse(::Type{ElT},
                         ::UndefInitializer,
                         boffs::BlockOffsets,
                         diaglength::Integer) where {ElT<:Number}
  return DiagBlockSparse(Vector{ElT}(undef,diaglength),boffs)
end

DiagBlockSparse(::UndefInitializer,
                boffs::BlockOffsets,
                diaglength::Integer) = DiagBlockSparse(Float64,undef,boffs,diaglength)

diagblockoffsets(D::DiagBlockSparse) = D.diagblockoffsets
blockoffsets(D::DiagBlockSparse) = D.diagblockoffsets

findblock(D::DiagBlockSparse{<:Number,<:Union{Number,AbstractVector},N},
          block::Block{N}; vargs...) where {N} = findblock(diagblockoffsets(D),block; vargs...)

const NonuniformDiagBlockSparse{ElT,VecT} = DiagBlockSparse{ElT,VecT} where {VecT<:AbstractVector}
const UniformDiagBlockSparse{ElT,VecT} = DiagBlockSparse{ElT,VecT} where {VecT<:Number}

Base.@propagate_inbounds Base.getindex(D::NonuniformDiagBlockSparse,i::Int)= data(D)[i]
Base.getindex(D::UniformDiagBlockSparse,i::Int) = data(D)

Base.@propagate_inbounds Base.setindex!(D::DiagBlockSparse,val,i::Int)= (data(D)[i] = val)
Base.setindex!(D::UniformDiagBlockSparse,val,i::Int)= error("Cannot set elements of a uniform DiagBlockSparse storage")

#Base.fill!(D::DiagBlockSparse,v) = fill!(data(D),v)

# convert to complex
# TODO: this could be a generic TensorStorage function
Base.complex(D::DiagBlockSparse) = DiagBlockSparse(complex(data(D)))

Base.copy(D::DiagBlockSparse) = DiagBlockSparse(copy(data(D)))

Base.conj(D::DiagBlockSparse{<:Real}) = D
Base.conj(D::DiagBlockSparse{<:Complex}) = DiagBlockSparse(conj(data(D)))

# TODO: make this generic for all storage types
Base.eltype(::DiagBlockSparse{ElT}) where {ElT} = ElT
Base.eltype(::Type{<:DiagBlockSparse{ElT}}) where {ElT} = ElT

# Deal with uniform DiagBlockSparse conversion
#Base.convert(::Type{<:DiagBlockSparse{ElT,VecT}},D::DiagBlockSparse) where {ElT,VecT} = DiagBlockSparse(convert(VecT,data(D)))

Base.size(D::DiagBlockSparse) = size(data(D))

# TODO: write in terms of ::Int, not inds
Base.similar(D::NonuniformDiagBlockSparse) = DiagBlockSparse(similar(data(D)))
#Base.similar(D::NonuniformDiagBlockSparse,inds) = DiagBlockSparse(similar(data(D),minimum(dims(inds))))
#function Base.similar(D::Type{<:NonuniformDiagBlockSparse{ElT,VecT}},inds) where {ElT,VecT}
#  return DiagBlockSparse(similar(VecT,diaglength(inds)))
#end

Base.similar(D::UniformDiagBlockSparse) = DiagBlockSparse(zero(T))
Base.similar(D::UniformDiagBlockSparse,inds) = similar(D)
Base.similar(::Type{<:UniformDiagBlockSparse{ElT}},inds) where {ElT} = DiagBlockSparse(zero(ElT))

Base.similar(D::DiagBlockSparse,n::Int) = DiagBlockSparse(similar(data(D),n))

Base.similar(D::DiagBlockSparse,::Type{ElR},n::Int) where {ElR} = DiagBlockSparse(similar(data(D),ElR,n))

# TODO: make this work for other storage besides Vector
Base.zeros(::Type{<:NonuniformDiagBlockSparse{ElT}},dim::Int64) where {ElT} = DiagBlockSparse(zeros(ElT,dim))
Base.zeros(::Type{<:UniformDiagBlockSparse{ElT}},dim::Int64) where {ElT} = DiagBlockSparse(zero(ElT))

Base.:*(D::DiagBlockSparse,x::Number) = DiagBlockSparse(x*data(D))
Base.:*(x::Number,D::DiagBlockSparse) = D*x

#
# Type promotions involving DiagBlockSparse
# Useful for knowing how conversions should work when adding and contracting
#

function Base.promote_rule(::Type{<:UniformDiagBlockSparse{ElT1}},
                           ::Type{<:UniformDiagBlockSparse{ElT2}}) where {ElT1,ElT2}
  ElR = promote_type(ElT1,ElT2)
  return DiagBlockSparse{ElR,ElR}
end

function Base.promote_rule(::Type{<:NonuniformDiagBlockSparse{ElT1,VecT1}},
                           ::Type{<:NonuniformDiagBlockSparse{ElT2,VecT2}}) where {ElT1,VecT1<:AbstractVector,
                                                                        ElT2,VecT2<:AbstractVector}
  ElR = promote_type(ElT1,ElT2)
  VecR = promote_type(VecT1,VecT2)
  return DiagBlockSparse{ElR,VecR}
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
function Base.promote_rule(::Type{<:UniformDiagBlockSparse{ElT1,VecT1}},
                           ::Type{<:NonuniformDiagBlockSparse{ElT2,Vector{ElT2}}}) where {ElT1,VecT1<:Number,
                                                                                          ElT2}
  ElR = promote_type(ElT1,ElT2)
  VecR = Vector{ElR}
  return DiagBlockSparse{ElR,VecR}
end

function Base.promote_rule(::Type{BlockSparseT1},
                           ::Type{<:NonuniformDiagBlockSparse{ElT2,VecT2}}) where {BlockSparseT1<:BlockSparse,
                                                                                   ElT2<:Number,VecT2<:AbstractVector}
  return promote_type(BlockSparseT1,BlockSparse{ElT2,VecT2})
end

function Base.promote_rule(::Type{BlockSparseT1},
                           ::Type{<:UniformDiagBlockSparse{ElT2,ElT2}}) where {BlockSparseT1<:BlockSparse,
                                                                        ElT2<:Number}
  return promote_type(BlockSparseT1,ElT2)
end

# Convert a DiagBlockSparse storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiagBlockSparse{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}
dense(::Type{<:UniformDiagBlockSparse{ElT}}) where {ElT} = Dense{ElT,Vector{ElT}}

const DiagBlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:DiagBlockSparse}
const NonuniformDiagBlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where 
                                               {StoreT<:NonuniformDiagBlockSparse}
const UniformDiagBlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where 
                                               {StoreT<:UniformDiagBlockSparse}

function DiagBlockSparseTensor(::Type{ElT},
                               ::UndefInitializer,
                               blocks::Blocks,
                               inds) where {ElT}
  blockoffsets,nnz = diagblockoffsets(blocks,inds)
  storage = DiagBlockSparse(ElT,undef,blockoffsets,nnz)
  return Tensor(storage,inds)
end

DiagBlockSparseTensor(::UndefInitializer,
                      blocks::Blocks,
                      inds) = DiagBlockSparseTensor(Float64,undef,blocks,inds)

function DiagBlockSparseTensor(::Type{ElT},
                               blocks::Blocks,
                               inds) where {ElT}
  blockoffsets,nnz = diagblockoffsets(blocks,inds)
  storage = DiagBlockSparse(ElT,blockoffsets,nnz)
  return Tensor(storage,inds)
end

DiagBlockSparseTensor(blocks::Blocks,
                      inds) = DiagBlockSparseTensor(Float64,blocks,inds)

# Uniform case
function DiagBlockSparseTensor(x::Number,
                               blocks::Blocks,
                               inds)
  blockoffsets,nnz = diagblockoffsets(blocks,inds)
  storage = DiagBlockSparse(x,blockoffsets)
  return Tensor(storage,inds)
end

diagblockoffsets(T::DiagBlockSparseTensor) = diagblockoffsets(store(T))

"""
blockview(T::DiagBlockSparseTensor,pos::Int)

Given a specified position in the block-offset list, return a Diag Tensor
that is a view to the data in that block (to avoid block lookup if the position
is known already).
"""
function blockview(T::DiagBlockSparseTensor,
                   pos::Int)
  blockT,offsetT = diagblockoffsets(T)[pos]
  return blockview(T,BlockOffset(blockT,offsetT))
end

function blockview(T::DiagBlockSparseTensor,
                   bof::BlockOffset)
  blockT,offsetT = bof
  blockdimsT = blockdims(T,blockT)
  blockdiaglengthT = minimum(blockdimsT)
  dataTslice = @view data(store(T))[offsetT+1:offsetT+blockdiaglengthT]
  return Tensor(Diag(dataTslice),blockdimsT)
end

function blockview(T::UniformDiagBlockSparseTensor,
                   bof::BlockOffset)
  blockT,offsetT = bof
  blockdimsT = blockdims(T,blockT)
  return Tensor(Diag(getdiagindex(T,1)),blockdimsT)
end

Base.IndexStyle(::Type{<:DiagBlockSparseTensor}) = IndexCartesian()

# TODO: this needs to be better (promote element type, check order compatibility,
# etc.
function Base.convert(::Type{<:DenseTensor{ElT,N}}, T::DiagBlockSparseTensor{ElT,N}) where {ElT<:Number,N}
  return dense(T)
end

# These are rules for determining the output of a pairwise contraction of Tensors
# (given the indices of the output tensors)
function contraction_output_type(TensorT1::Type{<:DiagBlockSparseTensor},
                                 TensorT2::Type{<:BlockSparseTensor},
                                 IndsR::Type)
  return similar_type(promote_type(TensorT1,TensorT2),IndsR)
end

contraction_output_type(TensorT1::Type{<:BlockSparseTensor},
                        TensorT2::Type{<:DiagBlockSparseTensor},
                        IndsR::Type) = contraction_output_type(TensorT2,TensorT1,IndsR)

# This performs the logic that DiagBlockSparseTensor*DiagBlockSparseTensor -> DiagBlockSparseTensor if it is not an outer
# product but -> DenseTensor if it is
# TODO: if the tensors are both order 2 (or less), or if there is an Index replacement,
# then they remain diagonal. Should we limit DiagBlockSparseTensor*DiagBlockSparseTensor to cases that
# result in a DiagBlockSparseTensor, for efficiency and type stability? What about a general
# SparseTensor result?
function contraction_output_type(TensorT1::Type{<:DiagBlockSparseTensor{<:Number,N1}},
                                 TensorT2::Type{<:DiagBlockSparseTensor{<:Number,N2}},
                                 IndsR::Type) where {N1,N2}
  if ValLength(IndsR)===Val{N1+N2}
    # Turn into is_outer(inds1,inds2,indsR) function?
    # How does type inference work with arithmatic of compile time values?
    return similar_type(dense(promote_type(TensorT1,TensorT2)),IndsR)
  end
  return similar_type(promote_type(TensorT1,TensorT2),IndsR)
end

# The output must be initialized as zero since it is sparse, cannot be undefined
contraction_output(T1::DiagBlockSparseTensor,T2::Tensor,indsR) = zero_contraction_output(T1,T2,indsR)
contraction_output(T1::Tensor,T2::DiagBlockSparseTensor,indsR) = contraction_output(T2,T1,indsR)

function contraction_output(T1::DiagBlockSparseTensor,
                            T2::DiagBlockSparseTensor,
                            indsR)
  return zero_contraction_output(T1,T2,indsR)
end

function array(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return array(dense(T))
end
matrix(T::DiagBlockSparseTensor{<:Number,2}) = array(T)
vector(T::DiagBlockSparseTensor{<:Number,1}) = array(T)

function Base.Array{ElT,N}(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return array(T)
end

function Base.Array(T::DiagBlockSparseTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

# Needed to get slice of DiagBlockSparseTensor like T[1:3,1:3]
function Base.similar(T::DiagBlockSparseTensor{<:Number,N},
                      ::Type{ElR},
                      inds::Dims{N}) where {ElR<:Number,N}
  return Tensor(similar(store(T),ElR,minimum(inds)),inds)
end

getdiagindex(T::DiagBlockSparseTensor{<:Number},ind::Int) = store(T)[ind]

setdiagindex!(T::DiagBlockSparseTensor,val,ind::Int) = (setindex!(T,val,ind); return T)

setdiag(T::DiagBlockSparseTensor,val,ind::Int) = Tensor(DiagBlockSparse(val),inds(T))

Base.@propagate_inbounds function Base.getindex(T::DiagBlockSparseTensor{ElT,N},
                                                inds::Vararg{Int,N}) where {ElT,N}
  if all(==(inds[1]),inds)
    return store(T)[inds[1]]
  else
    return zero(eltype(ElT))
  end
end
Base.@propagate_inbounds Base.getindex(T::DiagBlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]
Base.@propagate_inbounds Base.getindex(T::DiagBlockSparseTensor{<:Number,0}) = store(T)[1]

# Set diagonal elements
# Throw error for off-diagonal
Base.@propagate_inbounds function Base.setindex!(T::DiagBlockSparseTensor{<:Number,N},
                                                 val,inds::Vararg{Int,N}) where {N}
  all(==(inds[1]),inds) || error("Cannot set off-diagonal element of DiagBlockSparse storage")
  return store(T)[inds[1]] = val
end
Base.@propagate_inbounds Base.setindex!(T::DiagBlockSparseTensor{<:Number,1},val,ind::Int) = ( store(T)[ind] = val )
Base.@propagate_inbounds Base.setindex!(T::DiagBlockSparseTensor{<:Number,0},val) = ( store(T)[1] = val )

function Base.setindex!(T::UniformDiagBlockSparseTensor{<:Number,N},val,inds::Vararg{Int,N}) where {N}
  error("Cannot set elements of a uniform DiagBlockSparse storage")
end

# TODO: make a fill!! that works for uniform and non-uniform
#Base.fill!(T::DiagBlockSparseTensor,v) = fill!(store(T),v)

function dense(::Type{<:Tensor{ElT,N,StoreT,IndsT}}) where {ElT,N,
                                                            StoreT<:DiagBlockSparse,IndsT}
  return Tensor{ElT,N,dense(StoreT),IndsT}
end

# convert to Dense
function dense(T::TensorT) where {TensorT<:DiagBlockSparseTensor}
  R = zeros(dense(TensorT),inds(T))
  for i = 1:diaglength(T)
    setdiagindex!(R,getdiagindex(T,i),i)
  end
  return R
end

function outer!(R::DenseTensor{<:Number,NR},
                T1::DiagBlockSparseTensor{<:Number,N1},
                T2::DiagBlockSparseTensor{<:Number,N2}) where {NR,N1,N2}
  for i1 = 1:diaglength(T1), i2 = 1:diaglength(T2)
    indsR = CartesianIndex{NR}(ntuple(r -> r ≤ N1 ? i1 : i2, Val(NR)))
    R[indsR] = getdiagindex(T1,i1)*getdiagindex(T2,i2)
  end
  return R
end

# TODO: write an optimized version of this?
function outer!(R::DenseTensor{ElR},
                T1::DenseTensor,
                T2::DiagBlockSparseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R,T1,dense(T2))
  return R
end

function outer!(R::DenseTensor{ElR},
                T1::DiagBlockSparseTensor,
                T2::DenseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R,dense(T1),T2)
  return R
end

# Right an in-place version
function outer(T1::DiagBlockSparseTensor{ElT1,N1},
               T2::DiagBlockSparseTensor{ElT2,N2}) where {ElT1,ElT2,N1,N2}
  indsR = unioninds(inds(T1),inds(T2))
  R = Tensor(Dense(zeros(promote_type(ElT1,ElT2),dim(indsR))),indsR)
  outer!(R,T1,T2)
  return R
end

function Base.permutedims!(R::DiagBlockSparseTensor{<:Number,N},
                           T::DiagBlockSparseTensor{<:Number,N},
                           perm::NTuple{N,Int},f::Function=(r,t)->t) where {N}
  # TODO: check that inds(R)==permute(inds(T),perm)?
  for i=1:diaglength(R)
    @inbounds setdiagindex!(R,f(getdiagindex(R,i),getdiagindex(T,i)),i)
  end
  return R
end

function Base.permutedims(T::DiagBlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int},f::Function=identity) where {N}
  R = similar(T,permute(inds(T),perm))
  permutedims!(R,T,perm,f)
  return R
end

function Base.permutedims(T::UniformDiagBlockSparseTensor{ElT,N},
                          perm::NTuple{N,Int},
                          f::Function=identity) where {ElR,ElT,N}
  R = Tensor(DiagBlockSparse(f(getdiagindex(T,1))),permute(inds(T),perm))
  return R
end

# Version that may overwrite in-place or may return the result
function permutedims!!(R::NonuniformDiagBlockSparseTensor{<:Number,N},
                       T::NonuniformDiagBlockSparseTensor{<:Number,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {N}
  permutedims!(R,T,perm,f)
  return R
end

function permutedims!!(R::UniformDiagBlockSparseTensor{ElR,N},
                       T::UniformDiagBlockSparseTensor{ElT,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {ElR,ElT,N}
  R = Tensor(DiagBlockSparse(f(getdiagindex(R,1),getdiagindex(T,1))),inds(R))
  return R
end

function Base.permutedims!(R::DenseTensor{ElR,N},
                           T::DiagBlockSparseTensor{ElT,N},
                           perm::NTuple{N,Int},
                           f::Function = (r,t)->t) where {ElR,ElT,N}
  for i = 1:diaglength(T)
    @inbounds setdiagindex!(R,f(getdiagindex(R,i),getdiagindex(T,i)),i)
  end
  return R
end

function permutedims!!(R::DenseTensor{ElR,N},
                       T::DiagBlockSparseTensor{ElT,N},
                       perm::NTuple{N,Int},f::Function=(r,t)->t) where {ElR,ElT,N}
  permutedims!(R,T,perm,f)
  return R
end

function _contract!!(R::UniformDiagBlockSparseTensor{ElR,NR},labelsR,
                     T1::UniformDiagBlockSparseTensor{<:Number,N1},labelsT1,
                     T2::UniformDiagBlockSparseTensor{<:Number,N2},labelsT2) where {ElR,NR,N1,N2}
  if NR==0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    R = setdiag(R,diaglength(T1)*getdiagindex(T1,1)*getdiagindex(T2,1),1)
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    R = setdiag(R,getdiagindex(T1,1)*getdiagindex(T2,1),1)
  end
  return R
end

function contraction_output(T1::TensorT1,
                            labelsT1,
                            T2::TensorT2,
                            labelsT2,
                            labelsR) where {TensorT1<:BlockSparseTensor,
                                            TensorT2<:DiagBlockSparseTensor}
  indsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2,labelsR)
  TensorR = contraction_output_type(TensorT1,TensorT2,typeof(indsR))
  blockoffsetsR,contraction_plan = contract_blockoffsets(blockoffsets(T1),inds(T1),labelsT1,
                                                         blockoffsets(T2),inds(T2),labelsT2,
                                                         indsR,labelsR)
  R = zeros(TensorR,blockoffsetsR,indsR)
  return R,contraction_plan
end

function contract(T1::BlockSparseTensor,
                  labelsT1,
                  T2::DiagBlockSparseTensor,
                  labelsT2,
                  labelsR = contract_labels(labelsT1,labelsT2))
  R,contraction_plan = contraction_output(T1,labelsT1,T2,labelsT2,labelsR)
  R = contract!(R,labelsR,T1,labelsT1,T2,labelsT2,contraction_plan)
  return R
end

contract(T1::DiagBlockSparseTensor,
         labelsT1,
         T2::BlockSparseTensor,
         labelsT2,
         labelsR = contract_labels(labelsT2,labelsT1)) = contract(T2,labelsT2,T1,labelsT1,labelsR)

function contract!(R::BlockSparseTensor,
                   labelsR,
                   T1::BlockSparseTensor,
                   labelsT1,
                   T2::DiagBlockSparseTensor,
                   labelsT2,
                   contraction_plan)
  for (pos1,pos2,posR) in contraction_plan
    blockT1 = blockview(T1,pos1)
    blockT2 = blockview(T2,pos2)
    blockR = blockview(R,posR)
    contract!(blockR,labelsR,
              blockT1,labelsT1,
              blockT2,labelsT2)
  end
  return R
end

contract!(C::BlockSparseTensor,Clabels,
          A::BlockSparseTensor,Alabels,
          B::DiagBlockSparseTensor,Blabels) = contract!(C,Clabels,
                                                        B,Blabels,
                                                        A,Alabels)

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::DiagBlockSparseTensor)
  summary(io,T)
  println(io)
  for (block,_) in diagblockoffsets(T)
    blockdimsT = blockdims(T,block)
    # Print the location of the current block
    println(io,"Block: ",block)
    println(io," [",_range2string(blockstart(T,block),blockend(T,block)),"]")
    print_tensor(io,blockview(T,block))
    println(io)
    println(io)
  end
end

Base.show(io::IO, T::DiagBlockSparseTensor) = show(io,MIME("text/plain"),T)


