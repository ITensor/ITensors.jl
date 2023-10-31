function LinearAlgebra.qr(A::Exposed{<:MtlMatrix})
  Q, R = qr(expose(NDTensors.cpu(A)))
  return adapt(unwrap_type(A), Matrix(Q)), adapt(unwrap_type(A), R)
end

function NDTensors.Unwrap.qr_positive(A::Exposed{<:MtlMatrix})
  Q, R = qr_positive(expose(NDTensors.cpu(A)))
  return adapt(unwrap_type(A), Matrix(Q)), adapt(unwrap_type(A), R)
end

function NDTensors.Unwrap.ql(A::Exposed{<:MtlMatrix})
  Q, L = ql(expose(NDTensors.cpu(A)))
  return adapt(unwrap_type(A), Matrix(Q)), adapt(unwrap_type(A), L)
end
function NDTensors.Unwrap.ql_positive(A::Exposed{<:MtlMatrix})
  Q, L = ql_positive(expose(NDTensors.cpu(A)))
  return adapt(unwrap_type(A), Matrix(Q)), adapt(unwrap_type(A), L)
end

function LinearAlgebra.eigen(A::Exposed{<:MtlMatrix})
  D, U = eigen(expose(NDTensors.cpu(A)))
  return adapt(set_ndims(unwrap_type(A), ndims(D)), D), adapt(unwrap_type(A), U)
end

function LinearAlgebra.svd(A::Exposed{<:MtlMatrix})
  U, S, V = svd(expose(NDTensors.cpu(A)))
  return adapt(unwrap_type(A), U),
  adapt(set_ndims(unwrap_type(A), ndims(S)), S),
  adapt(unwrap_type(A), V)
end
