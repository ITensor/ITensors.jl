
@propagate_inbounds @inline function _setindex!!(
  ::HasQNs, T::Tensor, x::Number, I::Integer...
)
  fluxT = flux(T)
  setting_flux = isnothing(fluxT)
  if !NDTensors.is_unallocated_zeros(T) && fluxT != flux(T, I...)
    error(
      "In `setindex!`, the element $I of ITensor: \n$(T)\n you are trying to set is in a block with flux $(flux(T, I...)), which is different from the flux $fluxT of the other blocks of the ITensor. You may be trying to create an ITensor that does not have a well defined quantum number flux.",
    )
  end

  if setting_flux && NDTensors.is_unallocated_zeros(T)
    T = tensor(ITensor(eltype(T), flux(T, I...), inds(T)))
    T = setindex!!(T, x, I...)
    T = NDTensors.dropzeros(T; tol=zero(eltype(T)))
    return T
  end
  return setindex!!(T, x, I...)
end

"""
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds)
    ITensor([::Type{ElT} = Float64, ][flux::QN = QN(), ]inds::Index...)

Construct an ITensor with BlockSparse storage filled with `zero(ElT)`
where the nonzero blocks are determined by `flux`.

If `ElT` is not specified it defaults to `Float64`.

If `flux` is not specified, the ITensor will be empty (it will contain no blocks, and
have an undefined flux). The flux will be set by the first element that is set.

# Examples

```julia
julia> i
(dim=3|id=212|"i") <Out>
 1: QN(0) => 1
 2: QN(1) => 2

julia> @show ITensor(QN(0), i', dag(i));
ITensor(QN(0), i', dag(i)) = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{Float64, Vector{Float64}, 2}
 3×3
Block(1, 1)
 [1:1, 1:1]
 0.0

Block(2, 2)
 [2:3, 2:3]
 0.0  0.0
 0.0  0.0

julia> @show ITensor(QN(1), i', dag(i));
ITensor(QN(1), i', dag(i)) = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{Float64, Vector{Float64}, 2}
 3×3
Block(2, 1)
 [2:3, 1:1]
 0.0
 0.0

julia> @show ITensor(ComplexF64, QN(1), i', dag(i));
ITensor(ComplexF64, QN(1), i', dag(i)) = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{ComplexF64, Vector{ComplexF64}, 2}
 3×3
Block(2, 1)
 [2:3, 1:1]
 0.0 + 0.0im
 0.0 + 0.0im

julia> @show ITensor(undef, QN(1), i', dag(i));
ITensor(undef, QN(1), i', dag(i)) = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{Float64, Vector{Float64}, 2}
 3×3
Block(2, 1)
 [2:3, 1:1]
 0.0
 1.63e-322
```
Construction with undefined flux:
```julia
julia> A = ITensor(i', dag(i));

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.EmptyStorage{NDTensors.UnspecifiedZero, NDTensors.BlockSparse{NDTensors.UnspecifiedZero, Vector{NDTensors.UnspecifiedZero}, 2}}
 3×3



julia> isnothing(flux(A))
true

julia> A[i' => 1, i => 2] = 2
2

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=3|id=212|"i")' <Out>
 1: QN(0) => 1
 2: QN(1) => 2
Dim 2: (dim=3|id=212|"i") <In>
 1: QN(0) => 1
 2: QN(1) => 2
NDTensors.BlockSparse{Int64, Vector{Int64}, 2}
 3×3
Block(1, 2)
 [1:1, 2:3]
 2  0

julia> flux(A)
QN(-1)
```
"""
function ITensor(::Type{ElT}, flux::QN, inds::QNIndices) where {ElT<:Number}
  is = Tuple(inds)
  blocks = nzblocks(flux, is)
  if length(blocks) == 0
    ITensor(ElT, inds)
    #error("ITensor with flux=$flux resulted in no allowed blocks")
  end

  T = BlockSparseTensor(ITensors.default_datatype(ElT), blocks, is)
  return itensor(T)
end

# This helps with making code more generic between block sparse
# and dense.
function ITensor(::Type{ElT}, flux::QN, inds::Indices) where {ElT<:Number}
  return itensor(Dense(ElT, dim(inds)), inds)
end

function ITensor(::Type{ElT}, flux::QN, is...) where {ElT<:Number}
  return ITensor(ElT, flux, indices(is...))
end

ITensor(flux::QN, is...) = ITensor(ITensors.default_eltype(), flux, indices(is...))

function ITensor(::Type{ElT}, inds::QNIndices) where {ElT<:Number}
  is = Tuple(inds)
  T = BlockSparseTensor(ITensors.default_datatype(ElT), Vector{Block{0}}(), is)
  return itensor(T)
end

ITensor(inds::QNIndices) = ITensor(ITensors.default_eltype(), inds)

# TODO: generalize to list of Tuple, Vector, and QNIndex
function ITensor(::Type{ElT}, is::QNIndex...) where {ElT<:Number}
  return ITensor(ElT, QN(), indices(is...))
end

# TODO: generalize to list of Tuple, Vector, and QNIndex
ITensor(is::QNIndex...) = ITensor(ITensors.default_eltype(), indices(is))

"""
    ITensor([::Type{ElT} = Float64,] ::UndefInitializer, flux::QN, inds)
    ITensor([::Type{ElT} = Float64,] ::UndefInitializer, flux::QN, inds::Index...)

Construct an ITensor with indices `inds` and BlockSparse storage with undefined
elements of type `ElT`, where the nonzero (allocated) blocks are determined by
the provided QN `flux`. One purpose for using this constructor is that
initializing the elements in an undefined way is faster than initializing
them to a set value such as zero.

The storage will have `NDTensors.BlockSparse` type.

# Examples

```julia
i = Index([QN(0)=>1, QN(1)=>2], "i")
A = ITensor(undef,QN(0),i',dag(i))
B = ITensor(Float64,undef,QN(0),i',dag(i))
C = ITensor(ComplexF64,undef,QN(0),i',dag(i))
```
"""
function ITensor(
  ::Type{ElT}, ::UndefInitializer, flux::QN, inds::Indices
) where {ElT<:Number}
  is = Tuple(inds)
  blocks = nzblocks(flux, is)
  T = BlockSparseTensor(ElT, undef, blocks, is)
  return itensor(T)
end

function ITensor(::Type{ElT}, ::UndefInitializer, flux::QN, is...) where {ElT<:Number}
  return ITensor(ElT, undef, flux, indices(is...))
end

function ITensor(::UndefInitializer, flux::QN, is...)
  return ITensor(Float64, undef, flux, indices(is...))
end

"""
    ITensor([ElT::Type, ]x::Number, flux::QN, inds)
    ITensor([ElT::Type, ]x::Number, flux::QN, inds::Index...)

Construct an ITensor with all elements consistent with QN flux `flux` set to `x` and indices `inds`.

If `x isa Int` or `x isa Complex{Int}` then the elements will be set to `float(x)`
unless specified otherwise by the first input.

The storage will have `NDTensors.BlockSparse` type.

# Examples

```julia
i = Index([QN(0)=>1, QN(1)=>2], "i")
A = ITensor(2.3, QN(0), i', dag(i))
B = ITensor(Float64, 3.5, QN(0), i', dag(i))
C = ITensor(ComplexF64, 4, QN(0), i', dag(i))
```

!!! warning
    In future versions this may not automatically convert integer inputs with
    `float`, and in that case the particular element type should not be relied on.
"""
function ITensor(eltype::Type{<:Number}, x::Number, flux::QN, is::Indices)
  is_tuple = Tuple(is)
  blocks = nzblocks(flux, is_tuple)
  if length(blocks) == 0
    error("ITensor with flux=$flux resulted in no allowed blocks")
  end
  T = BlockSparseTensor(eltype(x), blocks, is_tuple)
  return itensor(T)
end

function ITensor(eltype::Type{<:Number}, x::Number, flux::QN, is...)
  return ITensor(eltype, x, flux, indices(is...))
end

ITensor(x::Number, flux::QN, is...) = ITensor(eltype(x), x, flux, is...)

ITensor(x::RealOrComplex{Int}, flux::QN, is...) = ITensor(float(x), flux, is...)

ITensor(eltype::Type{<:Number}, x::Number, is::QNIndices) = ITensor(eltype, x, QN(), is)

function ITensor(
  as::AliasStyle,
  elt::Type{<:Number},
  A::AbstractArray{<:Number},
  is::QNIndex,
  i...;
  kwargs...,
)
  tol = haskey(kwargs, :tol) ? kwargs[:tol] : 0.0
  checkflux = haskey(kwargs, :checkflux) ? kwargs[:checkflux] : true
  return QNITensor(as, elt, A, indices(is, i...))
end

"""
    ITensor([ElT::Type, ]::AbstractArray, inds; tol=0.0, checkflux=true)

Create a block sparse ITensor from the input Array, and collection
of QN indices. Zeros are dropped and nonzero blocks are determined
from the zero values of the array.

Optionally, you can set a tolerance such that elements
less than or equal to the tolerance are dropped.

By default, this will check that the flux of the nonzero blocks
are consistent with each other. You can disable this check by
setting `checkflux=false`.

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
function QNITensor(
  ::AliasStyle,
  elt::Type{<:Number},
  A::AbstractArray{<:Number},
  inds::Indices{<:QNIndex};
  tol=0.0,
  checkflux=true,
)
  is = Tuple(inds)
  length(A) ≠ dim(inds) && throw(
    DimensionMismatch(
      "In ITensor(::AbstractArray, inds), length of AbstractArray ($(length(A))) must match total dimension of the indices ($(dim(is)))",
    ),
  )
  blocks = Block{length(is)}[]
  T = BlockSparseTensor(elt, blocks, inds)
  A = reshape(A, dims(is)...)
  _copyto_dropzeros!(T, A; tol)
  if checkflux
    ITensors.checkflux(T)
  end
  return itensor(T)
end

function ITensor(
  as::AliasStyle,
  elt::Type{<:Number},
  A::AbstractArray{<:Number},
  inds::Indices{<:QNIndex};
  tol=0.0,
  checkflux=true,
)
  return ITensors.QNITensor(as, elt, A, inds; tol=tol, checkflux=checkflux)
end

function _copyto_dropzeros!(T::Tensor, A::AbstractArray; tol)
  for i in eachindex(T)
    Aᵢ = A[i]
    if abs(Aᵢ) > tol
      T[i] = Aᵢ
    end
  end
  return T
end

function combiner(inds::QNIndices; kwargs...)
  # TODO: support combining multiple set of indices
  is = Tuple(inds)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = ⊗(is...; kwargs...)
  new_ind = settags(new_ind, tags)
  comb_ind, perm, comb = combineblocks(new_ind)
  return itensor(Combiner(perm, comb), (comb_ind, dag.(is)...))
end

function dropzeros(T::ITensor; tol=0)
  # XXX: replace with empty(T)
  # T̃ = ITensor(zero(eltype(T)), inds(T))
  # @show data(T̃)
  # for b in eachnzblock(T)
  #   Tb = T[b]
  #   if norm(Tb) > tol
  #     T̃[b] = Tb
  #   end
  # end
  # return T̃
  t = NDTensors.dropzeros(tensor(T); tol = tol)
  return itensor(t)
end

function δ_split(i1::Index, i2::Index)
  d = emptyITensor(i1, i2)
  for n in 1:min(dim(i1), dim(i2))
    d[n, n] = 1
  end
  return d
end

function splitblocks(A::ITensor, is=inds(A); tol=0)
  if !hasqns(A)
    return A
  end
  isA = filterinds(A; inds=is)
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
  A = dropzeros(A; tol=tol)
  return A
end

function removeqn(T::ITensor, qn_name::String; mergeblocks=true)
  if !hasqns(T)
    return T
  end
  inds_R = removeqn(inds(T), qn_name; mergeblocks)
  R = ITensor(inds_R)
  for iv in eachindex(T)
    if !iszero(T[iv])
      R[iv] = T[iv]
    end
  end
  return R
end
