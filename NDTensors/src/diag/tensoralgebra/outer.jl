function outer!(
  R::DenseTensor{<:Number,NR}, T1::DiagTensor{<:Number,N1}, T2::DiagTensor{<:Number,N2}
) where {NR,N1,N2}
  for i1 in 1:diaglength(T1), i2 in 1:diaglength(T2)
    indsR = CartesianIndex{NR}(ntuple(r -> r ≤ N1 ? i1 : i2, Val(NR)))
    R[indsR] = getdiagindex(T1, i1) * getdiagindex(T2, i2)
  end
  return R
end

# TODO: write an optimized version of this?
function outer!(R::DenseTensor{ElR}, T1::DenseTensor, T2::DiagTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, T1, dense(T2))
  return R
end

function outer!(R::DenseTensor{ElR}, T1::DiagTensor, T2::DenseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, dense(T1), T2)
  return R
end

# Right an in-place version
function outer(T1::DiagTensor{ElT1,N1}, T2::DiagTensor{ElT2,N2}) where {ElT1,ElT2,N1,N2}
  indsR = unioninds(inds(T1), inds(T2))
  R = tensor(Dense(generic_zeros(promote_type(ElT1, ElT2), dim(indsR))), indsR)
  outer!(R, T1, T2)
  return R
end
