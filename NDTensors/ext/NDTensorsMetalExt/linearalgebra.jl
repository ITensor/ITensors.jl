function qr(A::Exposed{<:MtlArray})
  Q, R = qr(expose(NDTensors.cpu(unexpose(A))))
  return adapt(leaf_parenttype, Matrix(Q)), adapt(leaf_parenttype, R)
end

function eigen(A::Exposed{<:MtlArray})
  D, U = eigen(expose(NDTensors.cpu(unexpose(A))))
  return adapt(set_ndims(leaf_parenttype, ndims(D)), D), adapt(leaf_parenttype, U)
end

function svd(A::Exposed{<:MtlArray})
  U, S, V = svd(expose(NDTensors.cpu(unexpose(A))))
  return adapt(leaf_parenttype, U),
  adapt(set_ndims(leaf_parenttype, ndims(S)), S),
  adapt(leaf_parenttype, V)
end
