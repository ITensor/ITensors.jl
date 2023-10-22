function NDTensors.qr(leaf_parenttype::Type{<:MtlArray}, A::AbstractMatrix)
  Q, R = NDTensors.qr(NDTensors.cpu(A))
  return adapt(leaf_parenttype, Matrix(Q)), adapt(leaf_parenttype, R)
end

function NDTensors.eigen(leaf_parenttype::Type{<:MtlArray}, A::AbstractMatrix)
  D, U = NDTensors.eigen(NDTensors.cpu(A))
  return adapt(set_ndims(leaf_parenttype, ndims(D)), D), adapt(leaf_parenttype, U)
end

function NDTensors.svd(leaf_parenttype::Type{<:MtlArray}, A::AbstractMatrix)
  U, S, V = NDTensors.svd(NDTensors.cpu(A))
  return adapt(leaf_parenttype, U), adapt(set_ndims(leaf_parenttype, ndims(S)), S), adapt(leaf_parenttype, V)
end
