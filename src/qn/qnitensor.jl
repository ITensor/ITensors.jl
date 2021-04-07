
"""
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds)
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...)

Construct an ITensor with BlockSparse storage filled with `zero(ElT)` where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`.

# Examples

```julia
i = Index([QN(0)=>1, QN(1)=>2], "i")

# QN ITensors with flux of QN(0):

A = ITensor(i',dag(i))
B = ITensor(QN(0),i',dag(i))

# QN ITensor with flux of QN(1):

C = ITensor(QN(1),i',dag(i))

# Complex QN ITensor with flux of QN(1):

C = ITensor(ComplexF64,QN(1),i',dag(i))
```
"""
function ITensor(::Type{ElT}, flux::QN, inds::Indices) where {ElT <: Number}
  blocks = nzblocks(flux, IndexSet(inds))
  if length(blocks)==0
    error("ITensor with flux=$flux resulted in no allowed blocks")
  end
  T = BlockSparseTensor(ElT, blocks, inds)
  return itensor(T)
end

function ITensor(::Type{ElT}, flux::QN, inds::Index...) where {ElT <: Number}
  return ITensor(ElT, flux, IndexSet(inds...))
end

ITensor(flux::QN, inds::Indices) = ITensor(Float64, flux, inds)

ITensor(flux::QN, inds::Index...) =
  ITensor(Float64, flux, IndexSet(inds...))

ITensor(::Type{ElT}, inds::QNIndices) where {ElT <: Number} =
  ITensor(ElT, QN(), inds)

ITensor(inds::QNIndices) = ITensor(Float64, QN(), inds)

ITensor(::Type{ElT}, inds::QNIndex...) where {ElT<:Number} =
  ITensor(ElT, QN(), IndexSet(inds...))

ITensor(inds::QNIndex...) = ITensor(Float64, QN(), IndexSet(inds...))

"""
    ITensor([::Type{ElT} = Float64,] ::UndefInitializer, flux::QN, inds)
    ITensor([::Type{ElT} = Float64,] ::UndefInitializer, flux::QN, inds::Index...)

Construct an ITensor with indices `inds` and BlockSparse storage with undefined elements of type `ElT`, where the nonzero (allocated) blocks are determined by the provided QN `flux`. One purpose for using this constructor is that initializing the elements in an undefined way is faster than initializing them to a set value such as zero.

The storage will have `NDTensors.BlockSparse` type.

# Examples

```julia
i = Index([QN(0)=>1, QN(1)=>2], "i")
A = ITensor(undef,QN(0),i',dag(i))
B = ITensor(Float64,undef,QN(0),i',dag(i))
C = ITensor(ComplexF64,undef,QN(0),i',dag(i))
```
"""
function ITensor(::Type{ElT}, ::UndefInitializer, 
                 flux::QN, inds::Indices) where {ElT <: Number}
  blocks = nzblocks(flux, IndexSet(inds))
  T = BlockSparseTensor(ElT,undef,blocks,inds)
  return itensor(T)
end

function ITensor(::Type{ElT}, ::UndefInitializer, 
                 flux::QN, inds::Index...) where {ElT <: Number}
  return ITensor(ElT,undef,flux,IndexSet(inds...))
end

function ITensor(::UndefInitializer, 
                 flux::QN, inds::Index...)
  return ITensor(Float64,undef,flux,IndexSet(inds...))
end

"""
    ITensor([ElT::Type, ]::Array, inds; tol = 0)

Create a block sparse ITensor from the input Array, and collection 
of QN indices. Zeros are dropped and nonzero blocks are determined
from the zero values of the array.

Optionally, you can set a tolerance such that elements
less than or equal to the tolerance are dropped.

# Examples

```julia
julia> i = Index([QN(0)=>1, QN(1)=>2], "i");

julia> A = [1e-9 0.0 0.0;
            0.0 2.0 3.0;
            0.0 1e-10 4.0];

julia> @show ITensor(A, i', dag(i); tol = 1e-8);
ITensor(A, i', dag(i); tol = 1.0e-8) = ITensor ord=2
Dim 1: (dim=3|id=468|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=468|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{Float64,Array{Float64,1},2}
 3×3
Block: (2, 2)
 [2:3, 2:3]
 2.0  3.0
 0.0  4.0
```
"""
function itensor(::Type{ElT}, A::Array{<: Number}, inds::QNIndexSet; tol = 0) where {ElT <: Number}
  length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(::Array, ::IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  T = emptyITensor(ElT, inds)
  A = reshape(A, dims(inds))
  for vs in eachindex(T)
    Avs = A[vs]
    if abs(Avs) > tol
      T[vs] = A[vs]
    end
  end
  return T
end

# Short-circuit the non-QN version that does a copy here
function ITensor(::Type{ElT}, A::Array{ElT}, inds::QNIndexSet; kwargs...) where {ElT}
  return itensor(ElT, A, inds; kwargs...)
end

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
function randomITensor(::Type{ElT}, flux::QN, inds::Indices) where {ElT <: Number}
  T = ITensor(ElT, undef, flux, inds)
  randn!(T)
  return T
end

randomITensor(::Type{ElT}, flux::QN,
              inds::Index...) where {ElT<:Number} =
  randomITensor(ElT, flux, IndexSet(inds...))

function randomITensor(::Type{ElT},
                       inds::QNIndices) where {ElT<:Number}
  return randomITensor(ElT, QN(), inds)
end

randomITensor(flux::QN, inds::Indices) =
  randomITensor(Float64, flux, inds)

randomITensor(flux::QN, inds::Index...) =
  randomITensor(Float64, flux, IndexSet(inds...))

function randomITensor(::Type{ElT},
                       inds::QNIndex...) where {ElT<:Number}
  return randomITensor(ElT, QN(), IndexSet(inds...))
end

randomITensor(inds::QNIndices) =
  randomITensor(Float64, QN(), inds)

randomITensor(inds::QNIndex...) =
  randomITensor(Float64, QN(), IndexSet(inds...))

function combiner(inds::QNIndices; kwargs...)
  # TODO: support combining multiple set of indices
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = ⊗(inds...; kwargs...)
  new_ind = settags(new_ind,tags)
  comb_ind,perm,comb = combineblocks(new_ind)
  return itensor(Combiner(perm,comb), IndexSet(comb_ind, dag.(inds)...))
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

function diagITensor(x::ElT,
                     flux::QN,
                     is::QNIndices) where {ElT <: Number}
  blocks = nzdiagblocks(flux, IndexSet(is))
  T = DiagBlockSparseTensor(float(ElT), blocks, is)
  NDTensors.data(T) .= x
  return itensor(T)
end

function diagITensor(x::Number,
                     flux::QN,
                     is::Index...)
  return diagITensor(x, flux, IndexSet(is...))
end

diagITensor(x::Number, is::QNIndices) = diagITensor(x, QN(), is)

diagITensor(x::Number, is::QNIndex...) = diagITensor(x, IndexSet(is...))

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

function dropzeros(T::ITensor; tol = 0)
  # XXX: replace with empty(T)
  T̃ = emptyITensor(eltype(T), inds(T))
  for b in eachnzblock(T)
    Tb = T[b]
    if norm(Tb) > tol
      T̃[b] = Tb
    end
  end
  return T̃
end

function δ_split(i1::Index, i2::Index)
  d = emptyITensor(i1, i2)
  for n in 1:min(dim(i1), dim(i2))
    d[n, n] = 1
  end
  return d
end

function splitblocks(A::ITensor, is = inds(A); tol = 0)
  isA = filterinds(A; inds = is)
  for i in isA
    i_split = splitblocks(i)
    ĩ_split = sim(i_split)
    # Ideally use norm δ tensor but currently
    # it doesn't work properly:
    #A *= δ(dag(i), ĩ_split)
    d = δ_split(dag(i), ĩ_split)
    A *= δ_split(dag(i), ĩ_split)
    A = replaceind(A, ĩ_split, i_split)
  end
  A = dropzeros(A; tol = tol)
  return A
end

