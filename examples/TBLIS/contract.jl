using ITensors
using LinearAlgebra
using TBLIS

let
  d = 25

  nthreads = 4

  i = Index(d, "i")
  j = Index(d, "j")
  k = Index(d, "k")
  l = Index(d, "l")
  a = Index(d, "a")
  b = Index(d, "b")

  A = randomITensor(i, a, j, b)
  B = randomITensor(b, l, k, a)
  C = randomITensor(i, j, k, l)

  println("Contracting (d x d x d x d) tensors A * B -> C, d = ", d)
  @show inds(A)
  @show inds(B)
  @show inds(C)

  #
  # Use BLAS
  #

  ITensors.disable_tblis()
  BLAS.set_num_threads(nthreads)

  println()
  println("Using BLAS with $nthreads threads")

  C_blas = copy(C)
  time_blas = @belapsed $C_blas .= $A .* $B samples = 100

  println()
  println("Time (BLAS) = ", time_blas, " seconds")

  #
  # Use TBLIS
  #

  ITensors.enable_tblis()
  TBLIS.set_num_threads(4)

  println()
  println("Using TBLIS with $(TBLIS.get_num_threads()) threads")

  C_tblis = copy(C)
  time_tblis = @belapsed $C_tblis .= $A .* $B samples = 100

  println()
  println("Time (TBLIS) = ", time_tblis, " seconds")

  println()
  @show C_blas ≈ C_tblis

  println()
  println("Time (TBLIS) / Time (BLAS) = ", time_tblis / time_blas)
end
