
"""
    ITensor([::Type{ElT} = Float64, ]flux::QN, inds::IndexSet) where {ElT <: Number}

    ITensor([::Type{ElT} = Float64, ]flux::QN, inds::Index...) where {ElT <: Number}

Construct an ITensor with BlockSparse storage filled with `zero(ElT)` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`.
"""
function ITensor(::Type{ElT},
                 flux::QN,
                 inds::IndexSet) where {ElT <: Number}
  blocks = nzblocks(flux, inds)
  T = BlockSparseTensor(ElT, blocks, inds)
  return itensor(T)
end

function ITensor(::Type{ElT},
                 flux::QN,
                 inds::Index...) where {ElT <: Number}
  return ITensor(ElT, flux, IndexSet(inds...))
end

ITensor(flux::QN, inds::IndexSet) = ITensor(Float64, flux, inds)

ITensor(flux::QN,
        inds::Index...) = ITensor(Float64, flux, IndexSet(inds...))


ITensor(::Type{<:Number},
        inds::QNIndexSet) = error("Must specify flux")

ITensor(inds::QNIndexSet) = ITensor(Float64, inds)

function ITensor(::Type{ElT}, inds::QNIndex...) where {ElT<:Number} 
  return ITensor(ElT, IndexSet(inds...))
end

ITensor(inds::QNIndex...) = ITensor(Float64, IndexSet(inds...))

# TODO: bring back this version that creates an ITensor with no blocks?
#"""
#    ITensor([::Type{ElT} = Float64, ]inds::QNIndexSet) where {ElT <: Number}
#
#    ITensor([::Type{ElT} = Float64, ]inds::QNIndex...) where {ElT <: Number}
#
#Construct an ITensor with `NDTensors.BlockSparse` storage of element type `ElT` with the no blocks.
#
#If `ElT` is not specified it defaults to `Float64`.
#"""
#function ITensor(::Type{ElT}, inds::QNIndexSet) where {ElT<:Number} 
#  T = BlockSparseTensor(ElT, inds)
#  return itensor(T)
#end
#

"""
    randomITensor([::Type{ElT} = Float64, ]flux::QN, inds::IndexSet) where {ElT <: Number}

    randomITensor([::Type{ElT} = Float64, ]flux::QN, inds::Index...) where {ElT <: Number}

Construct an ITensor with `NDTensors.BlockSparse` storage filled with random elements of type `ElT` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`.
"""
function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::IndexSet) where {ElT<:Number}
  T = ITensor(ElT, flux, inds)
  randn!(T)
  return T
end

function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::Index...) where {ElT<:Number}
  return randomITensor(ElT, flux, IndexSet(inds...))
end

randomITensor(flux::QN,
              inds::IndexSet) = randomITensor(Float64, flux, inds)

randomITensor(flux::QN,
              inds::Index...) = randomITensor(Float64,
                                              flux,
                                              IndexSet(inds...))

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
  return itensor(Combiner(perm,comb),
                 IndexSet(comb_ind, dag.(inds)...))
end

combiner(inds::Tuple{Vararg{QNIndex}};
         kwargs...) = combiner(inds...; kwargs...)

#
# DiagBlockSparse ITensor constructors
#

"""
    diagITensor([::Type{ElT} = Float64, ]flux::QN, is::IndexSet)

    diagITensor([::Type{ElT} = Float64, ]flux::QN, is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with elements `zero(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`. If the element type is not specified, it defaults to `Float64`.
"""
function diagITensor(::Type{ElT},
                     flux::QN,
                     is::IndexSet{N}) where {ElT <: Number, N}
  blocks = nzdiagblocks(flux, is)
  T = DiagBlockSparseTensor(ElT, blocks, is)
  return itensor(T)
end

function diagITensor(::Type{ElT},
                     flux::QN,
                     inds::Index...) where {ElT <: Number}
  return diagITensor(ElT, flux, IndexSet(inds...))
end

diagITensor(::Type{<:Number},
            inds::QNIndexSet) = error("Must specify flux")

diagITensor(flux::QN,
            is::IndexSet) = diagITensor(Float64, flux, is)

diagITensor(flux::QN,
            inds::Index...) = diagITensor(Float64,
                                          flux,
                                          IndexSet(inds...))

diagITensor(inds::QNIndexSet) = error("Must specify flux")

"""
    delta([::Type{ElT} = Float64, ]flux::QN, is::IndexSet)

    delta([::Type{ElT} = Float64, ]flux::QN, is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with uniform elements `one(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`. If the element type is not specified, it defaults to `Float64`.
"""
function delta(::Type{ElT},
               flux::QN,
               inds::IndexSet) where {ElT <: Number}
  blocks = nzdiagblocks(flux, inds)
  T = DiagBlockSparseTensor(one(ElT), blocks, inds)
  return itensor(T)
end

function delta(::Type{ElT},
               flux::QN,
               inds::Index...) where {ElT <: Number}
  return delta(ElT, flux, IndexSet(inds...))
end

delta(flux::QN,
      inds::IndexSet) = delta(Float64, flux, is)

delta(flux::QN,
      inds::Index...) = delta(Float64, flux, IndexSet(inds...))

delta(::Type{<:Number},
      is::QNIndexSet) = error("Must specify flux")

#
# Possible constructors
#

#"""
#    diagITensor(v::Vector{T<:Number}, flux::QN, is::IndexSet)
#    diagITensor(v::Vector{T<:Number}, flux::QN, is::Index...)
#
#Make a sparse ITensor with non-zero elements only along the diagonal.
#The diagonal elements will be set to the values stored in `v` and
#the ITensor will have element type `float(T)`.
#The storage will have DiagBlockSparse type.
#"""
#function diagITensor(v::Vector{<:Number},
#                     flux::QN,
#                     is::IndexSet)
#  # TODO: check that the diagonal blocks all have the same flux
#  length(v) ≠ mindim(is) && error("Length of vector for diagonal must equal minimum of the dimension of the input indices")
#  return ITensor(DiagBlockSparse(float(v)),is)
#end
#
#function diagITensor(v::Vector{<:Number},
#                     flux::QN,
#                     inds::Index...)
#  return diagITensor(v,flux,IndexSet(inds...))
#end
#
#diagITensor(v::Vector{<:Number},
#            inds::QNIndexSet) = error("Must specify flux")

#"""
#    diagITensor(x::Number, flux::QN, is::IndexSet)
#    diagITensor(x::Number, flux::QN, is::Index...)
#
#Make a sparse ITensor with non-zero elements only along the diagonal. 
#The diagonal elements will be set to the value `x` and
#the ITensor will have element type `float(T)`.
#The storage will have DiagBlockSparse type.
#"""
#function diagITensor(x::Number,
#                     flux::QN,
#                     is::IndexSet)
#  return ITensor(Diag(fill(float(x),mindim(is))),is)
#end
#
#function diagITensor(x::Number,
#                     flux::QN,
#                     is::Index...)
#  return diagITensor(x,flux,IndexSet(is...))
#end
#
#diagITensor(x::Number, is::QNIndexSet) = error("Must specify flux")

