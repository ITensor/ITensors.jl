function promote_rule(
  storagetype1::Type{<:ArrayStorage}, storagetype2::Type{<:Diag}
)
  return promote_type(storagetype1, datatype(storagetype2))
end

function contraction_output_type(
  tensortype1::Type{<:DiagTensor}, tensortype2::Type{<:ArrayStorageTensor}, indsR
)
  return similartype(promote_type(tensortype1, tensortype2), indsR)
end

function contraction_output_type(
  tensortype1::Type{<:ArrayStorageTensor}, tensortype2::Type{<:DiagTensor}, indsR
)
  return contraction_output_type(tensortype2, tensortype1, indsR)
end

# TODO: Modernize this function, rewrite in terms of `Array` and `DiagonalArray`.
function contract!(
  C::ArrayStorageTensor{ElC,NC},
  Clabels,
  A::DiagTensor{ElA,NA},
  Alabels,
  B::ArrayStorageTensor{ElB,NB},
  Blabels,
  α::Number=one(ElC),
  β::Number=zero(ElC);
  convert_to_dense::Bool=true,
) where {ElA,NA,ElB,NB,ElC,NC}
  #@timeit_debug timer "diag-dense contract!" begin
  if all(i -> i < 0, Blabels)
    # If all of B is contracted
    # TODO: can also check NC+NB==NA
    min_dim = min(minimum(dims(A)), minimum(dims(B)))
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      # Assumes C starts set to 0
      c₁ = zero(ElC)
      for i in 1:min_dim
        c₁ += getdiagindex(A, i) * getdiagindex(B, i)
      end
      setdiagindex!(C, α * c₁ + β * getdiagindex(C, 1), 1)
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i in 1:min_dim
        setdiagindex!(
          C, α * getdiagindex(A, i) * getdiagindex(B, i) + β * getdiagindex(C, i), i
        )
      end
    end
  else
    # Most general contraction
    if convert_to_dense
      # TODO: Define a conversion from `DiagonalArray` to `Array`.
      contract!(C, Clabels, to_arraystorage(dense(A)), Alabels, B, Blabels, α, β)
    else
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
        c = zero(ElC)
        for j in 1:diaglength(A)
          # With α == 0 && β == 1
          C[cstart + j * c_cstride + coffset] +=
            getdiagindex(A, j) * B[bstart + j * b_cstride + boffset]
          # XXX: not sure if this is correct
          #C[cstart+j*c_cstride+coffset] += α * getdiagindex(A, j)* B[bstart+j*b_cstride+boffset] + β * C[cstart+j*c_cstride+coffset]
        end
      end
    end
  end
  #end # @timeit
end

function contract!(
  C::ArrayStorageTensor,
  Clabels,
  A::ArrayStorageTensor,
  Alabels,
  B::DiagTensor,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C)),
)
  return contract!(C, Clabels, B, Blabels, A, Alabels, α, β)
end
