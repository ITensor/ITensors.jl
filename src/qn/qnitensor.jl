
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

# TODO: make this default to QN()
ITensor(::Type{<:Number},
        inds::QNIndexSet) = error("Must specify flux")

ITensor(inds::QNIndexSet) = ITensor(Float64, inds)

function ITensor(::Type{ElT}, inds::QNIndex...) where {ElT<:Number} 
  return ITensor(ElT, IndexSet(inds...))
end

ITensor(inds::QNIndex...) = ITensor(Float64, IndexSet(inds...))

# TODO: make this use NDTensors.ZeroBlockSparse storage
"""
    zeroITensor([::Type{ElT} = Float64, ]inds::QNIndexSet) where {ElT <: Number}

    zeroITensor([::Type{ElT} = Float64, ]inds::QNIndex...) where {ElT <: Number}

Construct an ITensor with `NDTensors.BlockSparse` storage of element type `ElT` with the no blocks.

If `ElT` is not specified it defaults to `Float64`.

In the future, this will use the storage `NDTensors.ZeroBlockSparse`.
"""
function zeroITensor(::Type{ElT}, inds::QNIndexSet) where {ElT<:Number} 
  T = BlockSparseTensor(ElT, inds)
  return itensor(T)
end

zeroITensor(inds::QNIndexSet) = zeroITensor(Float64, inds)

"""
    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::IndexSet) where {ElT <: Number}

    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...) where {ElT <: Number}

Construct an ITensor with `NDTensors.BlockSparse` storage filled with random elements of type `ElT` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`. If the flux is not specified it defaults to `QN()`.
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

function randomITensor(::Type{ElT},
                       inds::QNIndexSet) where {ElT<:Number}
  return randomITensor(ElT, QN(), inds)
end

randomITensor(flux::QN,
              inds::IndexSet) = randomITensor(Float64, flux, inds)

randomITensor(flux::QN,
              inds::Index...) = randomITensor(Float64,
                                              flux,
                                              IndexSet(inds...))

function randomITensor(::Type{ElT},
                       inds::QNIndex...) where {ElT<:Number}
  return randomITensor(ElT, QN(), IndexSet(inds...))
end

randomITensor(inds::QNIndexSet) = randomITensor(Float64, QN(), inds)

randomITensor(inds::QNIndex...) = randomITensor(Float64, QN(), IndexSet(inds...))

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
    diagITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::IndexSet)

    diagITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with elements `zero(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`.

If the element type is not specified, it defaults to `Float64`. If theflux is not specified, it defaults to `QN()`.
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

diagITensor(flux::QN,
            is::IndexSet) = diagITensor(Float64, flux, is)

diagITensor(flux::QN,
            inds::Index...) = diagITensor(Float64,
                                          flux,
                                          IndexSet(inds...))

function diagITensor(::Type{ElT},
                     inds::QNIndexSet) where {ElT <: Number}
  return diagITensor(ElT, QN(), inds)
end

function diagITensor(inds::QNIndexSet)
  return diagITensor(Float64, QN(), inds)
end

"""
    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::IndexSet)

    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with uniform elements `one(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`.

If the element type is not specified, it defaults to `Float64`. If theflux is not specified, it defaults to `QN()`.
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

function delta(::Type{ElT},
               inds::QNIndexSet) where {ElT <: Number}
  return delta(ElT, QN(), inds)
end

delta(inds::QNIndexSet) = delta(Float64, QN(), inds)

