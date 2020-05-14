
"""
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds)
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...)

Construct an ITensor with BlockSparse storage filled with `zero(ElT)` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`.
"""
function ITensor(::Type{ElT},
                 flux::QN,
                 inds::Indices) where {ElT <: Number}
  blocks = nzblocks(flux, IndexSet(inds))
  T = BlockSparseTensor(ElT, blocks, inds)
  return itensor(T)
end

function ITensor(::Type{ElT},
                 flux::QN,
                 inds::Index...) where {ElT <: Number}
  return ITensor(ElT, flux, IndexSet(inds...))
end

ITensor(flux::QN, inds::Indices) = ITensor(Float64, flux, inds)

ITensor(flux::QN,
        inds::Index...) = ITensor(Float64, flux, IndexSet(inds...))

function ITensor(::Type{ElT},
                 inds::QNIndices) where {ElT <: Number}
  return ITensor(ElT, QN(), inds)
end

ITensor(inds::QNIndices) = ITensor(Float64, QN(), inds)

function ITensor(::Type{ElT}, inds::QNIndex...) where {ElT<:Number} 
  return ITensor(ElT, QN(), IndexSet(inds...))
end

ITensor(inds::QNIndex...) = ITensor(Float64, QN(), IndexSet(inds...))

"""
    emptyITensor([::Type{ElT} = Float64, ]inds)
    emptyITensor([::Type{ElT} = Float64, ]inds::QNIndex...)

Construct an ITensor with `NDTensors.BlockSparse` storage of element type `ElT` with the no blocks.

If `ElT` is not specified it defaults to `Float64`.

In the future, this will use the storage `NDTensors.EmptyBlockSparse`.
"""
function emptyITensor(::Type{ElT},
                      inds::QNIndices) where {ElT <: Number}
  return itensor(EmptyBlockSparseTensor(ElT, inds))
end

emptyITensor(inds::QNIndices) = emptyITensor(Float64, inds)

"""
    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds)
    randomITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...)

Construct an ITensor with `NDTensors.BlockSparse` storage filled with random elements of type `ElT` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`. If the flux is not specified it defaults to `QN()`.
"""
function randomITensor(::Type{ElT},
                       flux::QN,
                       inds::Indices) where {ElT <: Number}
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
                       inds::QNIndices) where {ElT<:Number}
  return randomITensor(ElT, QN(), inds)
end

randomITensor(flux::QN,
              inds::Indices) = randomITensor(Float64, flux, inds)

randomITensor(flux::QN,
              inds::Index...) = randomITensor(Float64,
                                              flux,
                                              IndexSet(inds...))

function randomITensor(::Type{ElT},
                       inds::QNIndex...) where {ElT<:Number}
  return randomITensor(ElT, QN(), IndexSet(inds...))
end

randomITensor(inds::QNIndices) = randomITensor(Float64, QN(), inds)

randomITensor(inds::QNIndex...) = randomITensor(Float64, QN(), IndexSet(inds...))

function combiner(inds::QNIndices; kwargs...)
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

#
# DiagBlockSparse ITensor constructors
#

"""
    diagITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]is)
    diagITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with elements `zero(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`.

If the element type is not specified, it defaults to `Float64`. If theflux is not specified, it defaults to `QN()`.
"""
function diagITensor(::Type{ElT},
                     flux::QN,
                     is::Indices) where {ElT <: Number}
  blocks = nzdiagblocks(flux, IndexSet(is))
  T = DiagBlockSparseTensor(ElT, blocks, is)
  return itensor(T)
end

function diagITensor(::Type{ElT},
                     flux::QN,
                     inds::Index...) where {ElT <: Number}
  return diagITensor(ElT, flux, IndexSet(inds...))
end

diagITensor(flux::QN,
            is::Indices) = diagITensor(Float64, flux, is)

diagITensor(flux::QN,
            inds::Index...) = diagITensor(Float64,
                                          flux,
                                          IndexSet(inds...))

function diagITensor(::Type{ElT},
                     inds::QNIndices) where {ElT <: Number}
  return diagITensor(ElT, QN(), inds)
end

function diagITensor(inds::QNIndices)
  return diagITensor(Float64, QN(), inds)
end

"""
    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is)
    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with uniform elements `one(ElT)`. The ITensor only has diagonal blocks consistent with the specified `flux`.

If the element type is not specified, it defaults to `Float64`. If theflux is not specified, it defaults to `QN()`.
"""
function delta(::Type{ElT},
               flux::QN,
               inds::Indices) where {ElT <: Number}
  blocks = nzdiagblocks(flux, IndexSet(inds))
  T = DiagBlockSparseTensor(one(ElT), blocks, inds)
  return itensor(T)
end

function delta(::Type{ElT},
               flux::QN,
               inds::Index...) where {ElT <: Number}
  return delta(ElT, flux, IndexSet(inds...))
end

delta(flux::QN,
      inds::Indices) = delta(Float64, flux, is)

delta(flux::QN,
      inds::Index...) = delta(Float64, flux, IndexSet(inds...))

function delta(::Type{ElT},
               inds::QNIndices) where {ElT <: Number}
  return delta(ElT, QN(), inds)
end

delta(inds::QNIndices) = delta(Float64, QN(), inds)

