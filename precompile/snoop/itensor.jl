let
  i = Index(2)
  A = randomITensor(i, i', i'')
  for T in (Float64, ComplexF64)
    A = randomITensor(T, i, i')
    B = randomITensor(T, i', i'')
    C = A * B
    U, S, V = svd(A, i)
    A, B = factorize(A, i)
  end
end
