function qr(A::ArrayStorageTensor; positive=false)
  if positive
    error("Not implemented")
  end
  Q, R = qr(storage(A))
  Q = convert(typeof(R), Q)
  i, j = inds(A)
  q = size(A, 1) < size(A, 2) ? i : j
  q = sim(q)
  Qₜ = tensor(Q, (i, q))
  Rₜ = tensor(R, (dag(q), j))
  return Qₜ, Rₜ
end
