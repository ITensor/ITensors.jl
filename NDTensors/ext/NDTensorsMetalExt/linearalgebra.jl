using NDTensors.TypeParameterAccessors: unwrap_array_type

function LinearAlgebra.qr(A::Exposed{<:MtlMatrix})
  Q, R = qr(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), R)
end

function NDTensors.Expose.qr_positive(A::Exposed{<:MtlMatrix})
  Q, R = qr_positive(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), R)
end

function NDTensors.Expose.ql(A::Exposed{<:MtlMatrix})
  Q, L = ql(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function NDTensors.Expose.ql_positive(A::Exposed{<:MtlMatrix})
  Q, L = ql_positive(expose(NDTensors.cpu(A)))
  return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end

using NDTensors.TypeParameterAccessors: set_type_parameters
function LinearAlgebra.eigen(A::Exposed{<:MtlMatrix})
  Dcpu, Ucpu = eigen(expose(NDTensors.cpu(A)))
  D = adapt(set_type_parameters(unwrap_array_type(A), (eltype, ndims), (eltype(Dcpu), ndims(Dcpu))), Dcpu)
  U = adapt(unwrap_array_type(A), Ucpu)
  return D, U
end

function LinearAlgebra.svd(A::Exposed{<:MtlMatrix}; kwargs...)
  Ucpu, Scpu, Vcpu = svd(expose(NDTensors.cpu(A)); kwargs...)
  U = adapt(unwrap_array_type(A), Ucpu)
  S = adapt(set_type_parameters(unwrap_array_type(A), (eltype, ndims), (eltype(Scpu), ndims(Scpu))), Scpu)
  V = adapt(unwrap_array_type(A), Vcpu)
  return U, S, V
end
