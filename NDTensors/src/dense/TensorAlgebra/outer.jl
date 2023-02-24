function outer!(
  R::DenseTensor{ElR}, T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}
) where {ElR,ElT1,ElT2}
  if ElT1 != ElT2
    # TODO: use promote instead
    # T1,T2 = promote(T1,T2)

    ElT1T2 = promote_type(ElT1, ElT2)
    if ElT1 != ElT1T2
      # TODO: get this working
      # T1 = ElR.(T1)
      T1 = one(ElT1T2) * T1
    end
    if ElT2 != ElT1T2
      # TODO: get this working
      # T2 = ElR.(T2)
      T2 = one(ElT1T2) * T2
    end
  end

  v1 = data(T1)
  v2 = data(T2)
  RM = reshape(R, length(v1), length(v2))
  #RM .= v1 .* transpose(v2)
  #mul!(RM, v1, transpose(v2))
  _gemm!('N', 'T', one(ElR), v1, v2, zero(ElR), RM)
  return R
end

# TODO: call outer!!, make this generic
function outer(T1::DenseTensor{ElT1}, T2::DenseTensor{ElT2}) where {ElT1,ElT2}
  array_outer = vec(array(T1)) * transpose(vec(array(T2)))
  inds_outer = unioninds(inds(T1), inds(T2))
  return tensor(Dense{promote_type(ElT1, ElT2)}(vec(array_outer)), inds_outer)
end
