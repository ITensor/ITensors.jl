using ITensors,
      Test

@testset "not" begin
  i = Index(2,"i")
  j = Index(2,"j")
  k = Index(2,"k")

  A = randomITensor(i, j, k')

  Ap = prime(A, not("j"))

  @test hassameinds(Ap, (i', j, k''))

  Ap = prime(A, tags = !"j")

  @test hassameinds(Ap, (i', j, k''))

  At = addtags(A, "x", not("k"))

  @test hassameinds(At, (addtags(i,"x"), addtags(j,"x"), k'))

  Ap2 = prime(A, 2, not(i))

  @test hassameinds(Ap2, (i, j'', k'''))

  Ap2 = prime(A, 2, inds = !i)

  @test hassameinds(Ap2, (i, j'', k'''))

  Ap3 = prime(A, 3, not(i, k'))

  @test hassameinds(Ap3, (i, j''', k'))

  Ap3 = prime(A, 3, !(i, k'))

  @test hassameinds(Ap3, (i, j''', k'))

  At2 = settags(A, "y", not(IndexSet(j, k')))

  @test hassameinds(At2, (settags(i, "y"), j, k'))

  At2 = settags(A, "y", inds = !IndexSet(j, k'))

  @test hassameinds(At2, (settags(i, "y"), j, k'))

  B = filterinds(A, plev = !0)

  @test hassameinds(B, (k',))
end

nothing
