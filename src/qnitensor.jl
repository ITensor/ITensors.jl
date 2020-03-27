
const QNIndexSet{N} = IndexSet{N,QNIndex}

function ITensor(::Type{ElT},
                 flux::QN,
                 inds::IndexSet) where {ElT<:Number}
  blocks = nzblocks(flux,inds)
  T = BlockSparseTensor(ElT,blocks,inds)
  return itensor(T)
end

function ITensor(inds::QNIndex...)
  T = BlockSparseTensor(IndexSet(inds))
  return itensor(T)
end

function ITensor(::Type{ELT},inds::QNIndex...) where {ELT<:Number} 
  T = BlockSparseTensor(ELT,inds)
  return itensor(T)
end

ITensor(::Type{T},
        flux::QN,
        inds::Index...) where {T<:Number} = ITensor(T,flux,IndexSet(inds...))


ITensor(flux::QN,inds::IndexSet) = ITensor(Float64,flux::QN,inds...)

ITensor(flux::QN,
        inds::Index...) = ITensor(flux,IndexSet(inds...))

function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::IndexSet) where {ElT<:Number}
  T = ITensor(ElT,flux,inds)
  randn!(T)
  return T
end

function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::Index...) where {ElT<:Number}
  return randomITensor(ElT,flux,IndexSet(inds...))
end

randomITensor(flux::QN,inds::IndexSet) = randomITensor(Float64,flux::QN,inds...)

randomITensor(flux::QN,
              inds::Index...) = randomITensor(flux,IndexSet(inds...))

# Throw error if flux is not specified
randomITensor(::Type{<:Number},
              inds::QNIndexSet) = error("In randomITensor constructor, must specify desired flux when using QN Indices")

function combiner(inds::QNIndex...; kwargs...)
  # TODO: support combining multiple set of indices
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = ⊗(inds...)
  if all(i->dir(i)!=Out,inds) && all(i->dir(i)!=In,inds)
    new_ind = dag(new_ind)
    new_ind = replaceqns(new_ind,-qnblocks(new_ind))
  end
  new_ind = settags(new_ind,tags)
  comb_ind,perm,comb = combineblocks(new_ind)
  return ITensor(Combiner(perm,comb),IndexSet(comb_ind,dag.(inds)...)),comb_ind
end
combiner(inds::Tuple{Vararg{QNIndex}}; kwargs...) = combiner(inds...; kwargs...)

#
# DiagBlockSparse ITensor constructors
#

"""
diagITensor(::Type{T}, flux::QN, is::IndexSet)

Make a sparse ITensor of element type T with non-zero elements
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
function diagITensor(::Type{ElT},
                     flux::QN,
                     is::IndexSet{N}) where {ElT<:Number,N}
  blocks = nzdiagblocks(flux,is)
  T = DiagBlockSparseTensor(ElT,blocks,is)
  return itensor(T)
end

diagITensor(::Type{<:Number},inds::QNIndexSet) = error("Must specify flux")

"""
diagITensor(::Type{T}, flux::QN, is::Index...)

Make a sparse ITensor of element type T with non-zero elements
only along the diagonal. Defaults to having `zero(T)` along the diagonal.
The storage will have Diag type.
"""
diagITensor(::Type{ElT},flux::QN,inds::QNIndex...) where {ElT<:Number} = diagITensor(ElT,flux,IndexSet(inds...))

"""
diagITensor(flux::QN, is::IndexSet)

Make a sparse ITensor of element type Float64 with non-zero elements
only along the diagonal. Defaults to storing zeros along the diagonal.
The storage will have Diag type.
"""
diagITensor(flux::QN, is::IndexSet) = diagITensor(Float64,flux,is)

"""
diagITensor(flux::QN,is::Index...)

Make a sparse ITensor of element type Float64 with non-zero elements 
only along the diagonal. Defaults to storing zeros along the diagonal.
The storage will have DiagBlockSparse type.
"""
diagITensor(flux::QN,inds::Index...) = diagITensor(flux,IndexSet(inds...))

diagITensor(inds::QNIndexSet) = error("Must specify flux")

"""
diagITensor(v::Vector{T<:Number},flux::QN,is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal.
The diagonal elements will be set to the values stored in `v` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlockSparse type.
"""
function diagITensor(v::Vector{<:Number},
                     flux::QN,
                     is::IndexSet)
  # TODO: check that the diagonal blocks all have the same flux
  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return ITensor(DiagBlockSparse(float(v)),is)
end

"""
diagITensor(v::Vector{T<:Number}, flux::QN, is::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal.
The diagonal elements will be set to the values stored in `v` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlockSparse type.
"""
function diagITensor(v::Vector{<:Number},
                     flux::QN,
                     inds::Index...)
  return diagITensor(v,flux,IndexSet(inds...))
end

diagITensor(v::Vector{<:Number},inds::QNIndexSet) = error("Must specify flux")

"""
diagITensor(x::Number, is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlockSparse type.
"""
function diagITensor(x::Number,
                     flux::QN,
                     is::IndexSet)
  return ITensor(Diag(fill(float(x),mindim(is))),is)
end

"""
diagITensor(x::Number, is::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlockSparse type.
"""
function diagITensor(x::Number,
                     flux::QN,
                     is::Index...)
  return diagITensor(x,flux,IndexSet(is...))
end

diagITensor(x::Number,is::QNIndexSet) = error("Must specify flux")

"""
    delta(::Type{T},inds::IndexSet)

Make a diagonal ITensor with all diagonal elements 1.
"""
function delta(::Type{ElT},is::QNIndexSet) where {ElT<:Number}
  blocks = nzdiagblocks(QN(),is)
  T = DiagBlockSparseTensor(one(ElT),blocks,is)
  return itensor(T)
end

function replaceindex!(A::ITensor,i::QNIndex,j::QNIndex)
  space(i) != space(j) && error("Indices must have the same spaces to be replaced")
  pos = indexpositions(A,i)
  isempty(pos) && error("Index not found")
  curdir = dir(inds(A)[pos[1]])
  j = setdir(j,curdir)
  return setinds!(A,setindex(inds(A),j,pos[1]))
end

flux(T::ITensor,vals::Int...) = flux(inds(T),vals...)

