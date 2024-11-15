using .SparseArraysBase: densearray
using .DiagonalArrays: DiagIndex, diaglength
using .TypeParameterAccessors: unwrap_array_type

# TODO: Move to a different file.
function promote_rule(
  storagetype1::Type{<:ArrayStorage}, storagetype2::Type{<:DiagonalArray}
)
  # TODO: Replace with `unwrap_array_type` once
  # https://github.com/ITensor/ITensors.jl/pull/1220
  # is merged.
  return promote_type(storagetype1, unwrap_array_type(storagetype2))
end

# TODO: Move to a different file.
function promote_rule(
  storagetype1::Type{<:DiagonalArray}, storagetype2::Type{<:DiagonalArray}
)
  return error("Not implemented yet")
end

function contraction_output_type(
  arraytype1::Type{<:DiagonalArray}, arraytype2::Type{<:DiagonalArray}, inds
)
  return error("Not implemented yet")
end

default_convert_to_dense() = true

# TODO: Modernize this function, rewrite in terms of `Array` and `DiagonalArray`.
# TODO: Move to `storage`.
function contract!(
  C::ArrayStorage,
  Clabels,
  A::DiagonalArray,
  Alabels,
  B::ArrayStorage,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C));
  convert_to_dense=default_convert_to_dense(),
)
  if convert_to_dense
    contract_dense!(C, Clabels, A, Alabels, B, Blabels, α, β)
    return C
  end
  if !isone(α) || !iszero(β)
    error(
      "`contract!(::ArrayStorageTensor, ::DiagTensor, ::ArrayStorageTensor, α, β; convert_to_dense = false)` with `α ≠ 1` or `β ≠ 0` is not currently supported. You can call it with `convert_to_dense = true` instead.",
    )
  end
  astarts = zeros(Int, length(Alabels))
  bstart = 0
  cstart = 0
  b_cstride = 0
  nbu = 0
  for ib in 1:length(Blabels)
    ia = findfirst(==(Blabels[ib]), Alabels)
    if !isnothing(ia)
      b_cstride += stride(B, ib)
      bstart += astarts[ia] * stride(B, ib)
    else
      nbu += 1
    end
  end
  c_cstride = 0
  for ic in 1:length(Clabels)
    ia = findfirst(==(Clabels[ic]), Alabels)
    if !isnothing(ia)
      c_cstride += stride(C, ic)
      cstart += astarts[ia] * stride(C, ic)
    end
  end
  # strides of the uncontracted dimensions of
  # B
  bustride = zeros(Int, nbu)
  custride = zeros(Int, nbu)
  # size of the uncontracted dimensions of
  # B, to be used in CartesianIndices
  busize = zeros(Int, nbu)
  n = 1
  for ib in 1:length(Blabels)
    if Blabels[ib] > 0
      bustride[n] = stride(B, ib)
      busize[n] = size(B, ib)
      ic = findfirst(==(Blabels[ib]), Clabels)
      custride[n] = stride(C, ic)
      n += 1
    end
  end
  boffset_orig = 1 - sum(strides(B))
  coffset_orig = 1 - sum(strides(C))
  cartesian_inds = CartesianIndices(Tuple(busize))
  for inds in cartesian_inds
    boffset = boffset_orig
    coffset = coffset_orig
    for i in 1:nbu
      ii = inds[i]
      boffset += ii * bustride[i]
      coffset += ii * custride[i]
    end
    c = zero(eltype(C))
    for j in 1:diaglength(A)
      # With α == 0 && β == 1
      C[cstart + j * c_cstride + coffset] +=
        A[DiagIndex(j)] * B[bstart + j * b_cstride + boffset]
      # XXX: not sure if this is correct
      #C[cstart+j*c_cstride+coffset] += α * A[DiagIndex(j)] * B[bstart+j*b_cstride+boffset] + β * C[cstart+j*c_cstride+coffset]
    end
  end
end

function contract!(
  C::ArrayStorage{<:Any,0},
  Clabels,
  A::DiagonalArray,
  Alabels,
  B::ArrayStorage,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C));
  convert_to_dense=nothing,
)
  # If all of B is contracted
  # TODO: can also check NC+NB==NA
  min_dim = min(minimum(size(A)), minimum(size(B)))
  if length(Clabels) == 0
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    # Assumes C starts set to 0
    c₁ = zero(eltype(C))
    for i in 1:min_dim
      c₁ += A[DiagIndex(i)] * B[DiagIndex(i)]
    end
    C[DiagIndex(1)] = α * c₁ + β * C[DiagIndex(1)]
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    # TODO: should we make this return a Diag storage?
    for i in 1:min_dim
      C[DiagIndex(i)] = α * A[DiagIndex(i)] * B[DiagIndex(i)] + β * C[DiagIndex(i)]
    end
  end
  return C
end

function contract_dense!(
  C::ArrayStorage,
  Clabels,
  A::DiagonalArray,
  Alabels,
  B::ArrayStorage,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C)),
)
  return contract!(C, Clabels, densearray(A), Alabels, B, Blabels, α, β)
end

# Overspecifying types to fix ambiguity error.
function contract!(
  C::ArrayStorage,
  Clabels,
  A::ArrayStorage,
  Alabels,
  B::DiagonalArray,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C));
  convert_to_dense=default_convert_to_dense(),
)
  return contract!(C, Clabels, B, Blabels, A, Alabels, α, β; convert_to_dense)
end
