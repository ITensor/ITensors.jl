using ITensors
using LinearAlgebra
using TBLIS

let
  χ = 500
  d = 3

  nthreads = 4

  l = Index(χ, "l")
  r = Index(χ, "r")
  s1 = Index(d, "s1")
  s2 = Index(d, "s2")
  h = Index(d, "h")

  indsA = (l, h, l')
  indsB = (s1, r, s2, l)
  indsC = (h, l', s1, r, s2)

  A = randomITensor(indsA)
  B = randomITensor(indsB)
  C = randomITensor(indsC)

  println("Contracting tensors A * B -> C, χ = $χ, d = $d")
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

  C_blas = copy(C)
  allocated_blas = @ballocated $C_blas .= $A .* $B samples = 100

  println()
  println("Time (BLAS) = ", time_blas, " seconds")
  println("Allocated (BLAS) = ", allocated_blas / 1024^2, " MiB")

  #
  # Use TBLIS
  #

  ITensors.enable_tblis()
  TBLIS.set_num_threads(4)

  println()
  println("Using TBLIS with $(TBLIS.get_num_threads()) threads")

  C_tblis = copy(C)
  time_tblis = @belapsed $C_tblis .= $A .* $B samples = 100

  C_tblis = copy(C)
  allocated_tblis = @ballocated $C_tblis .= $A .* $B samples = 100

  println()
  println("Time (TBLIS) = ", time_tblis, " seconds")
  println("Allocated (TBLIS) = ", allocated_tblis * 1e-6, " MiB")

  println()
  @show norm(C_blas - C_tblis)

  println()
  println("Time (TBLIS) / Time (BLAS) = ", time_tblis / time_blas)
end
