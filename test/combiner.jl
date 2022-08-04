using ITensors, Test
using Combinatorics: permutations

@testset "Combiner" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(4, "k")
  l = Index(5, "l")

  A = randomITensor(i, j, k, l)

  @testset "Basic combiner properties" begin
    C = combiner(i, j, k)
    @test eltype(storage(C)) === Number
    @test ITensors.data(C) isa NDTensors.NoData
    @test NDTensors.uncombinedinds(NDTensors.tensor(C)) == (i, j, k)
    C2 = copy(C)
    @test eltype(storage(C2)) === Number
    @test ITensors.data(C2) isa NDTensors.NoData
    @test NDTensors.uncombinedinds(NDTensors.tensor(C2)) == (i, j, k)
  end

  @testset "Empty combiner" begin
    C = combiner()
    @test order(C) == 0
    @test isnothing(combinedind(C))
    AC = A * C
    @test A == AC
    AC = C * A
    @test A == AC

    R = ITensor(0.0, j, l, k, i)
    R .= A .* C
    @test R == A

    R = ITensor(j, l, k, i)
    R .= A .* C
    @test R == A
  end

  @testset "Two index combiner" begin
    for inds_ij in permutations([i, j])
      C = combiner(inds_ij...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, l, k, c)
      @test c == commonind(B, C)
      @test combinedind(C) == c
      @test isnothing(combinedind(A))
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      @test isnothing(combinedind(D))
    end

    for inds_il in permutations([i, l])
      C = combiner(inds_il...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, j, k)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_ik in permutations([i, k])
      C = combiner(inds_ik...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, j, l)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_jk in permutations([j, k])
      C = combiner(inds_jk...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, i, l)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasinds(B, i, l)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_jl in permutations([j, l])
      C = combiner(inds_jl...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, i, k)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasinds(B, i, k)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_kl in permutations([k, l])
      C = combiner(inds_kl...)
      c = combinedind(C)
      B = A * C
      @test hasinds(B, i, j)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasinds(B, i, j)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end
  end

  @testset "Three index combiner" begin
    for inds_ijl in permutations([i, j, l])
      C = combiner(inds_ijl...)
      c = combinedind(C)
      B = A * C
      @test hasind(B, k)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasind(B, k)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_ijk in permutations([i, j, k])
      C = combiner(inds_ijk...)
      c = combinedind(C)
      B = A * C
      @test hasind(B, l)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasind(B, l)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end

    for inds_jkl in permutations([j, k, l])
      C = combiner(inds_jkl...)
      c = combinedind(C)
      B = A * C
      @test hasind(B, i)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      B = C * A
      @test hasind(B, i)
      @test c == commonind(B, C)
      D = B * C
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
      D = C * B
      @test hasinds(D, i, j, k, l)
      @test D ≈ A
    end
  end

  @testset "SVD/Combiner should play nice" begin
    C = combiner(i, j, k)
    c = combinedind(C)
    Ac = A * C
    U, S, V, spec, u, v = svd(Ac, c)
    Uc = C * U
    Ua, Sa, Va, spec, ua, va = svd(A, i, j, k)
    replaceind!(Ua, ua, u)
    @test A ≈ C * Ac
    @test A ≈ Ac * C
    @test Ua * C ≈ U
    @test C * Ua ≈ U
    @test Ua ≈ Uc
    @test Uc * S * V ≈ A
    @test (C * Ua) * S * V ≈ Ac
    C = combiner(i, j)
    c = combinedind(C)
    Ac = A * C
    U, S, V, spec, u, v = svd(Ac, c)
    Uc = U * C
    Ua, Sa, Va, spec, ua, va = svd(A, i, j)
    replaceind!(Ua, ua, u)
    @test Ua ≈ Uc
    @test Ua * C ≈ U
    @test C * Ua ≈ U
    @test Uc * S * V ≈ A
    @test (C * Ua) * S * V ≈ Ac
  end

  @testset "mult/Combiner should play nice" begin
    C = combiner(i, j, k)
    Ac = A * C
    B = randomITensor(l)
    AB = Ac * B
    @test AB * C ≈ A * B
  end

  @testset "Replace index combiner" begin
    C = combiner(l; tags="nl")
    c = combinedind(C)
    B = A * C
    replaceind!(B, c, l)
    @test B == A
  end
end

nothing
