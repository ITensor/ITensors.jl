# These are rules for determining the output of a pairwise contraction of NDTensors
# (given the indices of the output tensors)
function contraction_output_type(
  tensortype1::Type{<:DiagTensor}, tensortype2::Type{<:DenseTensor}, indsR
)
  return similartype(promote_type(tensortype1, tensortype2), indsR)
end
function contraction_output_type(
  tensortype1::Type{<:DenseTensor}, tensortype2::Type{<:DiagTensor}, indsR
)
  return contraction_output_type(tensortype2, tensortype1, indsR)
end

# This performs the logic that DiagTensor*DiagTensor -> DiagTensor if it is not an outer
# product but -> DenseTensor if it is
# TODO: if the tensors are both order 2 (or less), or if there is an Index replacement,
# then they remain diagonal. Should we limit DiagTensor*DiagTensor to cases that
# result in a DiagTensor, for efficiency and type stability? What about a general
# SparseTensor result?
function contraction_output_type(
  tensortype1::Type{<:DiagTensor}, tensortype2::Type{<:DiagTensor}, indsR
)
  if length(indsR) == ndims(tensortype1) + ndims(tensortype2)
    # Turn into is_outer(inds1,inds2,indsR) function?
    # How does type inference work with arithmatic of compile time values?
    return similartype(dense(promote_type(tensortype1, tensortype2)), indsR)
  end
  return similartype(promote_type(tensortype1, tensortype2), indsR)
end

# The output must be initialized as zero since it is sparse, cannot be undefined
function contraction_output(T1::DiagTensor, T2::Tensor, indsR)
  return zero_contraction_output(T1, T2, indsR)
end
contraction_output(T1::Tensor, T2::DiagTensor, indsR) = contraction_output(T2, T1, indsR)

function contraction_output(T1::DiagTensor, T2::DiagTensor, indsR)
  return zero_contraction_output(T1, T2, indsR)
end

function _contract!!(
  R::UniformDiagTensor{ElR,NR},
  labelsR,
  T1::UniformDiagTensor{<:Number,N1},
  labelsT1,
  T2::UniformDiagTensor{<:Number,N2},
  labelsT2,
) where {ElR,NR,N1,N2}
  if NR == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    R = setdiag(R, diaglength(T1) * getdiagindex(T1, 1) * getdiagindex(T2, 1))
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    R = setdiag(R, getdiagindex(T1, 1) * getdiagindex(T2, 1))
  end
  return R
end

function contract!(
  R::DiagTensor{ElR,NR},
  labelsR,
  T1::DiagTensor{<:Number,N1},
  labelsT1,
  T2::DiagTensor{<:Number,N2},
  labelsT2,
) where {ElR,NR,N1,N2}
  if NR == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    Rdiag = zero(ElR)
    for i in 1:diaglength(T1)
      Rdiag += getdiagindex(T1, i) * getdiagindex(T2, i)
    end
    setdiagindex!(R, Rdiag, 1)
  else
    min_dim = min(diaglength(T1), diaglength(T2))
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    for i in 1:min_dim
      setdiagindex!(R, getdiagindex(T1, i) * getdiagindex(T2, i), i)
    end
  end
  return R
end

function contract!(
  C::DenseTensor{ElC,NC},
  Clabels,
  A::DiagTensor{ElA,NA},
  Alabels,
  B::DenseTensor{ElB,NB},
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
      contract!(C, Clabels, dense(A), Alabels, B, Blabels, α, β)
    else
      if !isone(α) || !iszero(β)
        error(
          "`contract!(::DenseTensor, ::DiagTensor, ::DenseTensor, α, β; convert_to_dense = false)` with `α ≠ 1` or `β ≠ 0` is not currently supported. You can call it with `convert_to_dense = true` instead.",
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
  C::DenseTensor,
  Clabels,
  A::DenseTensor,
  Alabels,
  B::DiagTensor,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C)),
)
  return contract!(C, Clabels, B, Blabels, A, Alabels, α, β)
end
