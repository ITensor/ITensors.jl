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
  Dcpu, Ucpu = eigen(expose(NDTensors.cpu(A)))
  D = adapt(set_ndims(set_eltype(unwrap_type(A), eltype(Dcpu)), ndims(Dcpu)), Dcpu)
  U = adapt(unwrap_type(A), Ucpu)
  return D, U
end

function LinearAlgebra.svd(A::Exposed{<:MtlMatrix}; kwargs...)
  Ucpu, Scpu, Vcpu = svd(expose(NDTensors.cpu(A)); kwargs...)
  U = adapt(unwrap_type(A), Ucpu)
  S = adapt(set_ndims(set_eltype(unwrap_type(A), eltype(Scpu)), ndims(Scpu)), Scpu)
  V = adapt(unwrap_type(A), Vcpu)
  return U, S, V
end
