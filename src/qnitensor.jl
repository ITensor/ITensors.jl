
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

# Throw error if flux is not specified
randomITensor(::Type{ElT},
              inds::QNIndex...) where {ElT<:Number} = randomITensor(ElT,IndexSet(inds...))

randomITensor(inds::QNIndexSet) = randomITensor(Float64,inds)
randomITensor(inds::QNIndex...) = randomITensor(Float64,inds...)

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
# DiagBlock ITensor constructors
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
  # TODO: check that the diagonal blocks all have the same flux
  blocks = nzdiagblocks(flux,is)
  T = DiagBlockTensor(ElT,blocks,is)
  return itensor(T)
  #return ITensor{N}(DiagBlock(T,mindim(is)),is)
end

diagITensor(::Type{<:Number},inds::QNIndexSet) = error("Must specify flux")
diagITensor(::Type{ElT},inds::QNIndex...) where {ElT<:Number} = diagITensor(ElT,IndexSet(inds...))

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
The storage will have DiagBlock type.
"""
diagITensor(flux::QN,inds::Index...) = diagITensor(flux,IndexSet(inds...))

diagITensor(inds::QNIndexSet) = error("Must specify flux")
diagITensor(inds::QNIndex...) = diagITensor(IndexSet(inds...))

"""
diagITensor(v::Vector{T<:Number},flux::QN,is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal.
The diagonal elements will be set to the values stored in `v` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlock type.
"""
function diagITensor(v::Vector{<:Number},
                     flux::QN,
                     is::IndexSet)
  # TODO: check that the diagonal blocks all have the same flux
  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
  return ITensor(DiagBlock(float(v)),is)
end

"""
diagITensor(v::Vector{T<:Number}, flux::QN, is::Index...)

Make a sparse ITensor with non-zero elements only along the diagonal.
The diagonal elements will be set to the values stored in `v` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlock type.
"""
function diagITensor(v::Vector{<:Number},
                     flux::QN,
                     inds::Index...)
  return diagITensor(v,flux,IndexSet(inds...))
end

diagITensor(v::Vector{<:Number},inds::QNIndexSet) = error("Must specify flux")
diagITensor(v::Vector{<:Number},inds::QNIndex...) = diagITensor(v,IndexSet(inds...))

"""
diagITensor(x::Number, is::IndexSet)

Make a sparse ITensor with non-zero elements only along the diagonal. 
The diagonal elements will be set to the value `x` and
the ITensor will have element type `float(T)`.
The storage will have DiagBlock type.
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
The storage will have DiagBlock type.
"""
function diagITensor(x::Number,
                     flux::QN,
                     is::Index...)
  return diagITensor(x,flux,IndexSet(is...))
end

diagITensor(x::Number,is::QNIndexSet) = error("Must specify flux")
diagITensor(x::Number,inds::QNIndex...) = diagITensor(x,IndexSet(inds...))

