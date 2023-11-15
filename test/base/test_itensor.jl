using ITensors
using LinearAlgebra
using Test

using Combinatorics: permutations

import Random: seed!
import ITensors.NDTensors: DenseTensor

# Enable debug checking for these tests
ITensors.enable_debug_checks()

seed!(12345)

function invdigits(::Type{T}, x...) where {T}
  return T(sum([x[length(x) - k + 1] * 10^(k - 1) for k in 1:length(x)]))
end

@testset "Dense ITensor basic functionality" begin
  @testset "ITensor constructors" begin
    @testset "Default" begin
      A = ITensor()
      @test storage(A) isa NDTensors.EmptyStorage{NDTensors.EmptyNumber}
    end

    @testset "Undef with index" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = ITensor(undef, i)
      @test storage(A) isa NDTensors.Dense{Float64}
    end

    @testset "Default with indices" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = ITensor(i, j)
      @test storage(A) isa NDTensors.EmptyStorage{NDTensors.EmptyNumber}
    end

    @testset "diag" for ElType in (Float32, Float64, ComplexF32, ComplexF64)
      i, j = Index.(2, ("i", "j"))
      A = randomITensor(ElType, i, j)
      d = diag(A)
      @test d isa DenseTensor{ElType,1}
      @test d[1] == A[1, 1]
      @test d[2] == A[2, 2]
    end

    @testset "Index set operations" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = randomITensor(i, j)
      B = randomITensor(j, k)
      C = randomITensor(k, l)
      @test hascommoninds(A, B)
      @test hascommoninds(B, C)
      @test !hascommoninds(A, C)
    end

    @testset "isreal, iszero, real, imag" begin
      i, j = Index.(2, ("i", "j"))
      A = randomITensor(i, j)
      Ac = randomITensor(ComplexF64, i, j)
      Ar = real(Ac)
      Ai = imag(Ac)
      @test Ac ≈ Ar + im * Ai
      @test isreal(A)
      @test !isreal(Ac)
      @test isreal(Ar)
      @test isreal(Ai)
      @test !iszero(A)
      @test !iszero(real(A))
      @test iszero(imag(A))
      @test iszero(ITensor(0.0, i, j))
      @test iszero(ITensor(i, j))
    end

    @testset "map" begin
      A = randomITensor(Index(2))
      @test eltype(A) == Float64
      B = map(ComplexF64, A)
      @test B ≈ A
      @test eltype(B) == ComplexF64
      B = map(Float32, A)
      @test B ≈ A
      @test eltype(B) == Float32
      B = map(x -> 2x, A)
      @test B ≈ 2A
      @test eltype(B) == Float64
      @test_throws ErrorException map(x -> x + 1, A)
    end

    @testset "getindex with state string" begin
      i₁ = Index(2, "S=1/2")
      i₂ = Index(2, "S=1/2")
      v = ITensor(i₁, i₂)
      v[i₂ => "↑", i₁ => "↓"] = 1.0
      @test v[1, 1] == 0.0
      @test v[1, 2] == 0.0
      @test v[2, 1] == 1.0
      @test v[2, 2] == 0.0
      @test v[i₁ => "↑", i₂ => "↑"] == 0.0
      @test v[i₁ => "↑", i₂ => "↓"] == 0.0
      @test v[i₁ => "↓", i₂ => "↑"] == 1.0
      @test v[i₁ => "↓", i₂ => "↓"] == 0.0
    end

    @testset "getindex with state string" begin
      i₁ = Index(2, "S=1/2")
      i₂ = Index(2, "S=1/2")
      v = ITensor(i₁, i₂)
      v["↓", "↑"] = 1.0
      @test v[1, 1] == 0.0
      @test v[1, 2] == 0.0
      @test v[2, 1] == 1.0
      @test v[2, 2] == 0.0
      @test v["↑", "↑"] == 0.0
      @test v["↑", "↓"] == 0.0
      @test v["↓", "↑"] == 1.0
      @test v["↓", "↓"] == 0.0
    end

    @testset "getindex with end (lastindex, LastIndex)" begin
      a = Index(2)
      b = Index(3)
      A = randomITensor(a, b)
      @test A[end, end] == A[a => 2, b => 3]
      @test A[end - 1, end] == A[a => 1, b => 3]
      @test A[end - 1, end - 1] == A[a => 1, b => 2]
      @test A[end - 1, end - 2] == A[a => 1, b => 1]
      @test A[end - 1, 2 * (end - 2)] == A[a => 1, b => 2]
      @test A[2, end] == A[a => 2, b => 3]
      @test A[2, end - 1] == A[a => 2, b => 2]
      @test A[1, end] == A[a => 1, b => 3]
      @test A[1, end - 2] == A[a => 1, b => 1]
      @test A[end, 2] == A[a => 2, b => 2]
      @test A[end - 1, 2] == A[a => 1, b => 2]
      @test A[a => end, b => end] == A[a => 2, b => 3]
      @test A[a => end - 1, b => end] == A[a => 1, b => 3]
      @test A[a => end, b => end - 1] == A[a => 2, b => 2]
      @test A[a => end - 1, b => 2 * (end - 2)] == A[a => 1, b => 2]
      @test A[a => 2, b => end] == A[a => 2, b => 3]
      @test A[a => 2, b => end] == A[a => 2, b => 3]
      @test A[a => 1, b => end] == A[a => 1, b => 3]
      @test A[a => end, b => 3] == A[a => 2, b => 3]
      @test A[a => end, b => 2] == A[a => 2, b => 2]
      @test A[b => end, a => end] == A[a => 2, b => 3]
      @test A[b => end - 1, a => end] == A[a => 2, b => 2]
      @test A[b => end - 1, a => end - 1] == A[a => 1, b => 2]
      @test A[b => end - 2, a => end - 1] == A[a => 1, b => 1]
      @test A[b => 2 * (end - 2), a => end - 1] == A[a => 1, b => 2]
      @test A[b => 2, a => end] == A[a => 2, b => 2]
      @test A[b => 2, a => end - 1] == A[a => 1, b => 2]
      @test A[b => 1, a => end] == A[a => 2, b => 1]
      @test A[b => 1, a => end - 1] == A[a => 1, b => 1]
      @test A[b => end, a => 2] == A[a => 2, b => 3]
      @test A[b => end - 1, a => 2] == A[a => 2, b => 2]
      @test A[b => end, a => 1] == A[a => 1, b => 3]
      @test A[b => end - 2, a => 1] == A[a => 1, b => 1]
      @test A[b => end^2 - 7, a => 1] == A[a => 1, b => 2]

      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      B = randomITensor(i)
      @test B[i => end] == B[i => dim(i)]
      @test B[i => end - 1] == B[i => dim(i) - 1]
      @test B[end] == B[dim(i)]
      @test B[end - 1] == B[dim(i) - 1]
    end
    @testset "ITensor equality" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      Aij = randomITensor(i, j)
      Aji = permute(Aij, j, i)
      Bij′ = randomITensor(i, j')
      Cij′ = randomITensor(i, j')
      @test Aij == Aij
      @test Aij == Aji
      @test Bij′ != Cij′
      @test Bij′ != Aij
    end
    @testset "Set element with end (lastindex, LastIndex)" begin
      _i = Index(2, "i")
      _j = Index(3, "j")

      A = ITensor(_i, _j)
      A[_i => end, _j => end] = 2.5
      @test A[_i => dim(_i), _j => dim(_j)] == 2.5

      A = ITensor(_i, _j)
      A[_j => end, _i => end] = 3.5
      @test A[_i => dim(_i), _j => dim(_j)] == 3.5

      A = ITensor(_i, _j)
      A[_j => end, _i => 1] = 4.5
      @test A[_i => 1, _j => dim(_j)] == 4.5

      A = ITensor(_i, _j)
      A[_j => end - 1, _i => 1] = 4.5
      @test A[_i => 1, _j => dim(_j) - 1] == 4.5
    end

    @testset "Random" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = randomITensor(i, j)

      # Test hasind, hasinds
      @test hasind(A, i)
      @test hasind(i)(A)

      @test hasinds(A, i)
      @test hasinds(A, j)
      @test hasinds(A, [i, j])
      @test hasinds([i, j])(A)
      @test hasinds(A, IndexSet(j))
      @test hasinds(A, j, i)
      @test hasinds(A, (i, j))
      @test hasinds(A, IndexSet(i, j))
      @test hasinds(j, i)(A)
      @test hasinds(i)(A)
      @test hasinds(IndexSet(j))(A)
      @test hasinds((i, j))(A)
      @test hasinds(IndexSet(i, j))(A)

      @test storage(A) isa NDTensors.Dense{Float64}

      @test ndims(A) == order(A) == 2 == length(inds(A))
      @test Order(A) == Order(2)
      @test size(A) == dims(A) == (2, 2)
      @test dim(A) == 4

      At = randomITensor(Index(2), Index(3))
      @test maxdim(At) == 3
      @test mindim(At) == 2
      @test dim(At, 1) == 2
      @test dim(At, 2) == 3

      B = randomITensor(IndexSet(i, j))
      @test storage(B) isa NDTensors.Dense{Float64}
      @test ndims(B) == order(B) == 2 == length(inds(B))
      @test size(B) == dims(B) == (2, 2)

      A = randomITensor()
      @test eltype(A) == Float64
      @test ndims(A) == 0
    end

    @testset "trace (tr)" begin
      i, j, k, l = Index.((2, 3, 4, 5), ("i", "j", "k", "l"))
      T = randomITensor(j, k', i', k, j', i)
      trT1 = tr(T)
      trT2 = (T * δ(i, i') * δ(j, j') * δ(k, k'))[]
      @test trT1 ≈ trT2

      T = randomITensor(j, k', i', l, k, j', i)
      trT1 = tr(T)
      trT2 = T * δ(i, i') * δ(j, j') * δ(k, k')
      @test trT1 ≈ trT2
    end

    @testset "ITensor iteration" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = randomITensor(i, j)
      Is = eachindex(A)
      @test length(Is) == dim(A)
      sumA = 0.0
      for I in Is
        sumA += A[I]
      end
      @test sumA ≈ sum(ITensors.data(A))
      sumA = 0.0
      for a in A
        sumA += a
      end
      @test sumA ≈ sum(A)
      @test sumA ≈ sum(A)
    end

    @testset "From matrix" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      M = [1 2; 3 4]
      A = itensor(M, i, j)
      @test storage(A) isa NDTensors.Dense{Float64}

      @test M ≈ Matrix(A, i, j)
      @test M' ≈ Matrix(A, j, i)
      @test_throws DimensionMismatch vector(A)

      @test size(A, 1) == size(M, 1) == 2
      @test_throws BoundsError size(A, 3)
      @test_throws BoundsError size(A, 0)
      @test_throws ErrorException size(M, 0)
      # setstorage changes the internal data but not indices
      N = [5 6; 7 8]
      A = itensor(M, i, j)
      B = ITensors.setstorage(A, NDTensors.Dense(vec(N)))
      @test N == Matrix(B, i, j)
      @test storage(A) isa NDTensors.Dense{Float64}
      @test storage(B) isa NDTensors.Dense{Int}

      M = [1 2 3; 4 5 6]
      @test_throws DimensionMismatch itensor(M, i, j)
    end

    @testset "To Matrix" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      TM = randomITensor(i, j)

      M1 = matrix(TM)
      for ni in eachval(i), nj in eachval(j)
        @test M1[ni, nj] ≈ TM[i => ni, j => nj]
      end

      M2 = Matrix(TM, j, i)
      for ni in eachval(i), nj in eachval(j)
        @test M2[nj, ni] ≈ TM[i => ni, j => nj]
      end

      T3 = randomITensor(i, j, k)
      @test_throws DimensionMismatch Matrix(T3, i, j)
    end

    @testset "To Vector" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      TV = randomITensor(i)

      V = vector(TV)
      for ni in eachindval(i)
        @test V[val(ni)] ≈ TV[ni]
      end
      V = Vector(TV)
      for ni in eachindval(i)
        @test V[val(ni)] ≈ TV[ni]
      end
      V = Vector(TV, i)
      for ni in eachindval(i)
        @test V[val(ni)] ≈ TV[ni]
      end
      V = Vector{ComplexF64}(TV)
      for ni in eachindval(i)
        @test V[val(ni)] ≈ complex(TV[ni])
      end

      T2 = randomITensor(i, j)
      @test_throws DimensionMismatch vector(T2)
    end

    @testset "Complex" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = ITensor(Complex, i, j)
      @test storage(A) isa NDTensors.EmptyStorage{Complex}
    end

    @testset "Random complex" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      A = randomITensor(ComplexF64, i, j)
      @test storage(A) isa NDTensors.Dense{ComplexF64}
    end

    @testset "From complex matrix" begin
      i, j, k, l = Index.(2, ("i", "j", "k", "l"))
      M = [1+2im 2; 3 4]
      A = itensor(M, i, j)
      @test storage(A) isa NDTensors.Dense{ComplexF64}
    end
  end

  @testset "eltype promotion with scalar * and /" begin
    @test eltype(ITensor(1.0f0, Index(2)) * 2) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) .* 2) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) / 2) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) ./ 2) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) * 2.0f0) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) .* 2.0f0) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) / 2.0f0) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) ./ 2.0f0) === Float32
    @test eltype(ITensor(1.0f0, Index(2)) * 2.0) === Float64
    @test eltype(ITensor(1.0f0, Index(2)) .* 2.0) === Float64
    @test eltype(ITensor(1.0f0, Index(2)) / 2.0) === Float64
    @test eltype(ITensor(1.0f0, Index(2)) ./ 2.0) === Float64
  end

  @testset "Division /" begin
    i = Index(2)
    A = randomITensor(i)
    B = A / 2
    C = A / ITensor(2)
    @test B isa ITensor
    @test C isa ITensor
    @test B ≈ C
    @test A[1] / 2 ≈ B[1]
    @test A[2] / 2 ≈ B[2]
    @test A[1] / 2 ≈ C[1]
    @test A[2] / 2 ≈ C[2]
  end

  @testset "Convert to complex" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, j)
    B = complex(A)
    for ii in 1:dim(i), jj in 1:dim(j)
      @test complex(A[i => ii, j => jj]) == B[i => ii, j => jj]
    end
  end

  @testset "Complex Number Operations" for _eltype in (Float32, Float64)
    i = Index(3, "i")
    j = Index(4, "j")

    A = randomITensor(complex(_eltype), i, j)

    rA = real(A)
    iA = imag(A)
    @test norm(rA + 1im * iA - A) < 1E-8
    @test eltype(rA) <: _eltype
    @test eltype(iA) <: _eltype

    cA = conj(A)
    @test eltype(cA) <: complex(_eltype)
    @test norm(cA) ≈ norm(A)

    B = randomITensor(_eltype, i, j)

    cB = conj(B)
    @test eltype(cB) <: _eltype
    @test norm(cB) ≈ norm(B)
  end

  @testset "similar" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, j)
    B = similar(A)
    @test inds(B) == inds(A)
    Ac = similar(A, ComplexF32)
    @test storage(Ac) isa NDTensors.Dense{ComplexF32}
  end

  @testset "fill!" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, j)
    fill!(A, 1.0)
    @test all(ITensors.data(A) .== 1.0)
  end

  @testset "fill! using broadcast" begin
    i = Index(2, "i")
    j = Index(2, "j")
    A = randomITensor(i, j)
    A .= 1.0
    @test all(ITensors.data(A) .== 1.0)
  end

  @testset "zero" begin
    i = Index(2)
    A = randomITensor(i)
    B = zero(A)
    @test false * A ≈ B
  end

  @testset "copyto!" begin
    i = Index(2, "i")
    j = Index(2, "j")
    M = [1 2; 3 4]
    A = itensor(M, i, j)
    N = 2 * M
    B = itensor(N, i, j)
    copyto!(A, B)
    @test A == B
    @test ITensors.data(A) == vec(N)
    A = itensor(M, i, j)
    B = itensor(N, j, i)
    copyto!(A, B)
    @test A == B
    @test ITensors.data(A) == vec(transpose(N))
  end

  @testset "Unary -" begin
    i = Index(2, "i")
    j = Index(2, "j")
    M = [1 2; 3 4]
    A = itensor(M, i, j)
    @test -A == itensor(-M, i, j)
  end

  @testset "dot" begin
    i = Index(2, "i")
    a = [1.0; 2.0]
    b = [3.0; 4.0]
    A = itensor(a, i)
    B = itensor(b, i)
    @test dot(A, B) == 11.0
  end

  @testset "mul!" begin
    i = Index(2; tags="i")
    j = Index(2; tags="j")
    k = Index(2; tags="k")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(i, k)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, i)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(i, j)
    B = randomITensor(k, j)
    C = randomITensor(i, k)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(i, j)
    B = randomITensor(k, j)
    C = randomITensor(k, i)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(j, i)
    B = randomITensor(j, k)
    C = randomITensor(i, k)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(j, i)
    B = randomITensor(j, k)
    C = randomITensor(k, i)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(j, i)
    B = randomITensor(k, j)
    C = randomITensor(i, k)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(j, i)
    B = randomITensor(k, j)
    C = randomITensor(k, i)
    mul!(C, A, B)
    @test C ≈ A * B

    A = randomITensor(i, j)
    B = randomITensor(k, j)
    C = randomITensor(k, i)
    α = 2
    β = 3
    R = mul!(copy(C), A, B, α, β)
    @test α * A * B + β * C ≈ R

    @testset "In-place bugs" begin
      @testset "Bug 1" begin
        l1 = Index(3, "l=1")
        l2 = Index(3, "l=2")
        s = Index(2, "s")

        A = randomITensor(s', s)
        B = randomITensor(l1, s, l2)

        C = randomITensor(l1, s', l2)

        C .= A .* B

        @test C ≈ A * B
      end

      @testset "Bug 2" begin
        is = [Index(n + 1, "i$n") for n in 1:6]

        for ais in permutations((1, 2, 3)),
          bis in permutations((2, 3, 4)),
          cis in permutations((1, 4))

          A = randomITensor(ntuple(i -> is[ais[i]], Val(length(ais))))
          B = randomITensor(ntuple(i -> is[bis[i]], Val(length(bis))))
          C = randomITensor(ntuple(i -> is[cis[i]], Val(length(cis))))

          C .= A .* B

          @test C ≈ A * B
        end

        for ais in permutations((1, 2, 3)),
          bis in permutations((2, 3, 4, 5)),
          cis in permutations((1, 4, 5))

          A = randomITensor(ntuple(i -> is[ais[i]], Val(length(ais))))
          B = randomITensor(ntuple(i -> is[bis[i]], Val(length(bis))))
          C = randomITensor(ntuple(i -> is[cis[i]], Val(length(cis))))

          C .= A .* B

          @test C ≈ A * B
        end
      end
    end

    @testset "In-place outer bug" begin
      l1 = Index(3, "l=1")
      s = Index(2, "s")

      A = randomITensor(l1)
      B = randomITensor(s)
      C = randomITensor(s, l1)

      C .= A .* B

      @test C ≈ A * B
    end

    @testset "In-place contractions" begin
      i1 = Index(2, "i1")
      i2 = Index(2, "i2")
      i3 = Index(2, "i3")
      i4 = Index(2, "i4")
      i5 = Index(2, "i5")
      i6 = Index(2, "i6")
      j1 = Index(2, "j1")
      j2 = Index(2, "j2")
      j3 = Index(2, "j3")

      #A = randomITensor(s', s)
      #B = randomITensor(l1, s, l2)

      #C = randomITensor(l1, s', l2)

      C .= A .* B
      @test C ≈ A * B
    end
  end

  @testset "exponentiate" begin
    s1 = Index(2, "s1")
    s2 = Index(2, "s2")
    i1 = Index(2, "i1")
    i2 = Index(2, "i2")
    Amat = rand(2, 2, 2, 2)
    A = itensor(Amat, i1, i2, s1, s2)

    Aexp = exp(A, (i1, i2), (s1, s2))
    Amatexp = reshape(exp(reshape(Amat, 4, 4)), 2, 2, 2, 2)
    Aexp_from_mat = itensor(Amatexp, i1, i2, s1, s2)
    @test Aexp ≈ Aexp_from_mat

    #test that exponentiation works when indices need to be permuted
    Aexp = exp(A, (s1, s2), (i1, i2))
    Amatexp = Matrix(exp(reshape(Amat, 4, 4))')
    Aexp_from_mat = itensor(reshape(Amatexp, 2, 2, 2, 2), s1, s2, i1, i2)
    @test Aexp ≈ Aexp_from_mat

    #test exponentiation when hermitian=true is used
    Amat = reshape(Amat, 4, 4)
    Amat = reshape(Amat + Amat' + randn(4, 4) * 1e-10, 2, 2, 2, 2)
    A = itensor(Amat, i1, i2, s1, s2)
    Aexp = exp(A, (i1, i2), (s1, s2); ishermitian=true)
    Amatexp = reshape(parent(exp(LinearAlgebra.Hermitian(reshape(Amat, 4, 4)))), 2, 2, 2, 2)
    Aexp_from_mat = itensor(Amatexp, i1, i2, s1, s2)
    @test Aexp ≈ Aexp_from_mat
    Aexp = exp(A, (i1, i2), (s1, s2); ishermitian=true)
    Amatexp = reshape(parent(exp(LinearAlgebra.Hermitian(reshape(Amat, 4, 4)))), 2, 2, 2, 2)
    Aexp_from_mat = itensor(Amatexp, i1, i2, s1, s2)
    @test Aexp ≈ Aexp_from_mat
  end

  @testset "onehot (setelt)" begin
    i = Index(2, "i")

    T = onehot(i => 1)
    @test eltype(T) === Float64
    @test T[i => 1] ≈ 1.0
    @test T[i => 2] ≈ 0.0

    T = setelt(i => 2)
    @test T[i => 1] ≈ 0.0
    @test T[i => 2] ≈ 1.0

    j = Index(2, "j")

    T = onehot(j => 2, i => 1)
    @test T[j => 1, i => 1] ≈ 0.0
    @test T[j => 2, i => 1] ≈ 1.0
    @test T[j => 1, i => 2] ≈ 0.0
    @test T[j => 2, i => 2] ≈ 0.0

    T = onehot(Float32, i => 1)
    @test eltype(T) === Float32
    @test T[i => 1] ≈ 1.0
    @test T[i => 2] ≈ 0.0

    T = onehot(ComplexF32, i => 1)
    @test eltype(T) === ComplexF32
    @test T[i => 1] ≈ 1.0
    @test T[i => 2] ≈ 0.0
  end

  @testset "add, subtract, and axpy" begin
    i = Index(2, "i")
    a = [1.0; 2.0]
    b = [3.0; 4.0]
    A = itensor(a, i)
    B = itensor(b, i)
    c = [5.0; 8.0]
    @test A + B == itensor([4.0; 6.0], i)
    @test axpy!(2.0, A, B) == itensor(c, i)
    a = [1.0; 2.0]
    b = [3.0; 4.0]
    A = itensor(a, i)
    B = itensor(b, i)
    c = [5.0; 8.0]
    @test (B .+= 2.0 .* A) == itensor(c, i)
    a = [1.0; 2.0]
    b = [3.0; 4.0]
    A = itensor(a, i)
    B = itensor(b, i)
    c = [8.0; 12.0]
    @test (A .= 2.0 .* A .+ 2.0 .* B) == itensor(c, i)
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(2.0)
    @test_throws DimensionMismatch A + B
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor()
    C = A + B
    @test C ≈ A
    A[1] = 5
    @test C[1] == 5
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(0)
    @test_throws DimensionMismatch A + B
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(ComplexF64)
    @test_throws DimensionMismatch A + B
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(Float64)
    @test_throws DimensionMismatch A + B
    a = [1.0; 2.0]
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(2.0)
    @test_throws DimensionMismatch A - B
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor()
    C = A - B
    @test C ≈ A
    A[1] = 5
    @test C[1] == 5
    #@test_throws DimensionMismatch A - B
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(2.0)
    @test_throws DimensionMismatch B - A
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor(Float64)
    @test_throws DimensionMismatch B - A
    a = [1.0; 2.0]
    A = itensor(a, i)
    B = ITensor()
    C = B - A
    @test C ≈ -A
    A[1] = 5
    @test C[1] == -1
    a = [1.0; 2.0]
    b = [3.0; 4.0]
    A = itensor(a, i)
    B = itensor(b, i)
    c = [2.0; 2.0]
    @test B - A == itensor(c, i)
    @test A - B == -itensor(c, i)
  end

  @testset "mul! and rmul!" begin
    i = Index(2, "i")
    a = [1.0; 2.0]
    b = [2.0; 4.0]
    A = itensor(a, i)
    A2, A3 = copy(A), copy(A)
    B = itensor(b, i)
    @test mul!(A2, A, 2.0) == B == (A2 .= 0 .* A2 .+ 2 .* A)
    @test rmul!(A, 2.0) == B == ITensors.scale!(A3, 2)
    #make sure mul! works also when A2 has NaNs in it
    A = itensor([1.0; 2.0], i)
    A2 = itensor([NaN; 1.0], i)
    @test mul!(A2, A, 2.0) == B

    i = Index(2, "i")
    j = Index(2, "j")
    M = [1 2; 3 4]
    A = itensor(M, i, j)
    N = 2 * M
    B = itensor(N, j, i)
    @test ITensors.data(mul!(B, A, 2.0)) == 2.0 * vec(transpose(M))
  end

  @testset "Construct from Array" begin
    i = Index(2, "index_i")
    j = Index(2, "index_j")

    M = [
      1.0 2
      3 4
    ]
    T = itensor(M, i, j)
    T[i => 1, j => 1] = 3.3
    @test M[1, 1] == 3.3
    @test T[i => 1, j => 1] == 3.3
    @test storage(T) isa NDTensors.Dense{Float64}

    M = [
      1 2
      3 4
    ]
    T = itensor(M, i, j)
    T[i => 1, j => 1] = 3.3
    @test M[1, 1] == 1
    @test T[i => 1, j => 1] == 3.3
    @test storage(T) isa NDTensors.Dense{Float64}

    M = [
      1 2
      3 4
    ]
    T = itensor(Int, M, i, j)
    T[i => 1, j => 1] = 6
    @test M[1, 1] == 6
    @test T[i => 1, j => 1] == 6
    @test storage(T) isa NDTensors.Dense{Int}

    # This version makes a copy
    M = [
      1.0 2
      3 4
    ]
    T = ITensor(M, i, j)
    T[i => 1, j => 1] = 3.3
    @test M[1, 1] == 1
    @test T[i => 1, j => 1] == 3.3

    # Empty indices
    A = randn(1)
    T = itensor(A, Index[])
    @test A[] == T[]
    T = itensor(A, Index[], Index[])
    @test A[] == T[]
    T = itensor(A, Any[])
    @test A[] == T[]

    A = randn(1, 1)
    T = itensor(A, Index[])
    @test A[] == T[]
    T = itensor(A, Index[], Index[])
    @test A[] == T[]
    T = itensor(A, Any[], Any[])
    @test A[] == T[]

    @test_throws ErrorException itensor(rand(1), Int[1])
  end

  @testset "Construct from AbstractArray" begin
    i = Index(2, "index_i")
    j = Index(2, "index_j")

    X = [
      1.0 2 0
      3 4 0
      0 0 0
    ]
    M = @view X[1:2, 1:2]
    T = itensor(M, i, j)
    T[i => 1, j => 1] = 3.3
    @test M[1, 1] == 3.3
    @test T[i => 1, j => 1] == 3.3
    @test storage(T) isa NDTensors.Dense{Float64}
  end

  @testset "ITensor Array constructor view behavior" begin
    d = 2
    i = Index(d)

    # view
    A = randn(Float64, d, d)
    T = itensor(A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Float64}
    A[1, 1] = 2.0
    T[1, 1] == 2.0

    # view
    A = rand(Int, d, d)
    T = itensor(Int, A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Int}
    A[1, 1] = 2
    T[1, 1] == 2

    # no view
    A = rand(Int, d, d)
    T = itensor(A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Float64}
    A[1, 1] = 2
    T[1, 1] ≠ 2

    # no view
    A = randn(Float64, d, d)
    T = ITensor(A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Float64}
    A[1, 1] = 2
    T[1, 1] ≠ 2

    # no view
    A = rand(Int, d, d)
    T = ITensor(Int, A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Int}
    A[1, 1] = 2
    T[1, 1] ≠ 2

    # no view
    A = rand(Int, d, d)
    T = ITensor(A, i', dag(i))
    @test storage(T) isa NDTensors.Dense{Float64}
    A[1, 1] = 2
    T[1, 1] ≠ 2
  end

  @testset "Convert to Array" begin
    i = Index(2, "i")
    j = Index(3, "j")
    T = randomITensor(i, j)

    A = Array{Float64}(T, i, j)
    for I in CartesianIndices(T)
      @test A[I] == T[I]
    end

    T11 = T[1, 1]
    T[1, 1] = 1
    @test T[1, 1] == 1
    @test T11 != 1
    @test A[1, 1] == T11

    A = Matrix{Float64}(T, i, j)
    for I in CartesianIndices(T)
      @test A[I] == T[I]
    end

    A = Matrix(T, i, j)
    for I in CartesianIndices(T)
      @test A[I] == T[I]
    end

    A = Array(T, i, j)
    for I in CartesianIndices(T)
      @test A[I] == T[I]
    end

    T = randomITensor(i)
    A = Vector(T)
    for I in CartesianIndices(T)
      @test A[I] == T[I]
    end
  end

  @testset "Test isapprox for ITensors" begin
    m, n = rand(0:20, 2)
    i = Index(m)
    j = Index(n)
    realData = rand(m, n)
    complexData = complex(realData)
    A = itensor(realData, i, j)
    B = itensor(complexData, i, j)
    @test A ≈ B
    @test B ≈ A
    A = permute(A, j, i)
    @test A ≈ B
    @test B ≈ A
  end

  @testset "permute" begin
    i = Index(2)
    A = ITensor(i, i')
    Ap = permute(A, i, i')
    A[i => 1, i' => 1] = 1
    @test A[i => 1, i' => 1] == 1
    @test Ap[i => 1, i' => 1] == 0
  end

  @testset "permute, NeverAlias()/AllowAlias()" begin
    i = Index(2)
    A = ITensor(i, i')
    Ap = permute(A, i, i')
    A[i => 1, i' => 1] = 1
    @test A[i => 1, i' => 1] == 1
    @test Ap[i => 1, i' => 1] == 0

    i = Index(2)
    A = ITensor(i, i')
    Ap = permute(ITensors.NeverAlias(), A, i, i')
    A[i => 1, i' => 1] = 1
    @test A[i => 1, i' => 1] == 1
    @test Ap[i => 1, i' => 1] == 0

    i = Index(2, "index_i")
    j = Index(4, "index_j")
    k = Index(3, "index_k")
    T = randomITensor(i, j, k)

    # NeverAlias()/allow_alias = false by default
    pT_noalias_1 = permute(T, i, j, k)
    pT_noalias_1[1, 1, 1] = 12
    @test T[1, 1, 1] != pT_noalias_1[1, 1, 1]

    pT_noalias_2 = permute(T, i, j, k; allow_alias=false)
    pT_noalias_2[1, 1, 1] = 12
    @test T[1, 1, 1] != pT_noalias_1[1, 1, 1]

    cT = copy(T)
    pT_alias = permute(cT, i, j, k; allow_alias=true)
    pT_alias[1, 1, 1] = 12
    @test cT[1, 1, 1] == pT_alias[1, 1, 1]

    cT = copy(T)
    pT_alias = permute(ITensors.AllowAlias(), cT, i, j, k)
    pT_alias[1, 1, 1] = 12
    @test cT[1, 1, 1] == pT_alias[1, 1, 1]
  end

  @testset "ITensor tagging and priming" begin
    s1 = Index(2, "Site,s=1")
    s2 = Index(2, "Site,s=2")
    l = Index(3, "Link")
    ltmp = settags(l, "Temp")
    A1 = randomITensor(s1, l, l')
    A2 = randomITensor(s2, l', l'')
    @testset "firstind(::ITensor,::String)" begin
      @test s1 == firstind(A1, "Site")
      @test s1 == firstind(A1, "s=1")
      @test s1 == firstind(A1, "s=1,Site")
      @test l == firstind(A1; tags="Link", plev=0)
      @test l' == firstind(A1; plev=1)
      @test l' == firstind(A1; tags="Link", plev=1)
      @test s2 == firstind(A2, "Site")
      @test s2 == firstind(A2, "s=2")
      @test s2 == firstind(A2, "Site")
      @test s2 == firstind(A2; plev=0)
      @test s2 == firstind(A2; tags="s=2", plev=0)
      @test s2 == firstind(A2; tags="Site", plev=0)
      @test s2 == firstind(A2; tags="s=2,Site", plev=0)
      @test l' == firstind(A2; plev=1)
      @test l' == firstind(A2; tags="Link", plev=1)
      @test l'' == firstind(A2; plev=2)
      @test l'' == firstind(A2; tags="Link", plev=2)
    end
    @testset "addtags(::ITensor,::String,::String)" begin
      s1u = addtags(s1, "u")
      lu = addtags(l, "u")

      A1u = addtags(A1, "u")
      @test hasinds(A1u, s1u, lu, lu')

      A1u = addtags(A1, "u", "Link")
      @test hasinds(A1u, s1, lu, lu')

      A1u = addtags(A1, "u"; tags="Link")
      @test hasinds(A1u, s1, lu, lu')

      A1u = addtags(A1, "u"; plev=0)
      @test hasinds(A1u, s1u, lu, l')

      A1u = addtags(A1, "u"; tags="Link", plev=0)
      @test hasinds(A1u, s1, lu, l')

      A1u = addtags(A1, "u"; tags="Link", plev=1)
      @test hasinds(A1u, s1, l, lu')
    end
    @testset "removetags(::ITensor,::String,::String)" begin
      A2r = removetags(A2, "Site")
      @test hasinds(A2r, removetags(s2, "Site"), l', l'')

      A2r = removetags(A2, "Link"; plev=1)
      @test hasinds(A2r, s2, removetags(l, "Link")', l'')

      A2r = replacetags(A2, "Link", "Temp"; plev=1)
      @test hasinds(A2r, s2, ltmp', l'')
    end
    @testset "replacetags(::ITensor,::String,::String)" begin
      s2tmp = replacetags(s2, "Site", "Temp")

      @test s2tmp == replacetags(s2, "Site" => "Temp")

      ltmp = replacetags(l, "Link", "Temp")

      A2r = replacetags(A2, "Site", "Temp")
      @test hasinds(A2r, s2tmp, l', l'')

      A2r = replacetags(A2, "Site" => "Temp")
      @test hasinds(A2r, s2tmp, l', l'')

      A2r = replacetags(A2, "Link", "Temp")
      @test hasinds(A2r, s2, ltmp', ltmp'')

      A2r = replacetags(A2, "Site" => "Link", "Link" => "Site")
      @test hasinds(
        A2r,
        replacetags(s2, "Site" => "Link"),
        replacetags(l', "Link" => "Site"),
        replacetags(l'', "Link" => "Site"),
      )
    end
    @testset "prime(::ITensor,::String)" begin
      A2p = prime(A2)
      @test A2p == A2'
      @test hasinds(A2p, s2', l'', l''')

      A2p = prime(A2, 2)
      A2p = A2''
      @test hasinds(A2p, s2'', l''', l'''')

      A2p = prime(A2, "s=2")
      @test hasinds(A2p, s2', l', l'')
    end

    @testset "mapprime" begin
      @test hasinds(mapprime(A2, 1, 7), s2, l^7, l'')
      @test hasinds(mapprime(A2, 0, 1), s2', l', l'')
    end

    @testset "replaceprime" begin
      @test hasinds(mapprime(A2, 1 => 7), s2, l^7, l'')
      @test hasinds(mapprime(A2, 0 => 1), s2', l', l'')
      @test hasinds(mapprime(A2, 1 => 7, 0 => 1), s2', l^7, l'')
      @test hasinds(mapprime(A2, 1 => 2, 2 => 1), s2, l'', l')
      @test hasinds(mapprime(A2, 1 => 0, 0 => 1), s2', l, l'')
    end

    @testset "setprime" begin
      @test hasinds(setprime(A2, 2, s2), s2'', l', l'')
      @test hasinds(setprime(A2, 0, l''), s2, l', l)
    end

    @testset "swapprime" begin
      @test hasinds(swapprime(A2, 1, 3), l''', s2, l'')
    end
  end

  @testset "ITensor other index operations" begin
    s1 = Index(2, "Site,s=1")
    s2 = Index(2, "Site,s=2")
    l = Index(3, "Link")
    A1 = randomITensor(s1, l, l')
    A2 = randomITensor(s2, l', l'')

    @testset "ind(::ITensor)" begin
      @test ind(A1, 1) == s1
      @test ind(A1, 2) == l
    end

    @testset "replaceind and replaceinds" begin
      rA1 = replaceind(A1, s1, s2)
      @test hasinds(rA1, s2, l, l')
      @test hasinds(A1, s1, l, l')

      @test replaceinds(A1, [] => []) == A1
      @test replaceinds(A1, ()) == A1
      @test replaceinds(A1) == A1

      # Pair notation (like Julia's replace function)
      rA1 = replaceind(A1, s1 => s2)
      @test hasinds(rA1, s2, l, l')
      @test hasinds(A1, s1, l, l')

      replaceind!(A1, s1, s2)
      @test hasinds(A1, s2, l, l')

      rA2 = replaceinds(A2, (s2, l'), (s1, l))
      @test hasinds(rA2, s1, l, l'')
      @test hasinds(A2, s2, l', l'')

      # Pair notation (like Julia's replace function)
      rA2 = replaceinds(A2, s2 => s1, l' => l)
      @test hassameinds(rA2, (s1, l, l''))
      @test hassameinds(A2, (s2, l', l''))

      # Test ignoring indices that don't exist
      rA2 = replaceinds(A2, s1 => l, l' => l)
      @test hassameinds(rA2, (s2, l, l''))
      @test hassameinds(A2, (s2, l', l''))

      replaceinds!(A2, (s2, l'), (s1, l))
      @test hasinds(A2, s1, l, l'')
    end

    @testset "replaceinds fixed errors" begin
      l = Index(3; tags="l")
      s = Index(2; tags="s")
      l̃, s̃ = sim(l), sim(s)
      A = randomITensor(s, l)
      Ã = replaceinds(A, (l, s), (l̃, s̃))
      @test ind(A, 1) == s
      @test ind(A, 2) == l
      @test ind(Ã, 1) == s̃
      @test ind(Ã, 2) == l̃
      @test_throws ErrorException replaceinds(A, (l, s), (s̃, l̃))
    end

    @testset "swapinds and swapinds!" begin
      s = Index(2)
      t = Index(2)
      Ast = randomITensor(s, s', t, t')
      Ats = swapinds(Ast, (s, s'), (t, t'))
      @test Ast != Ats
      @test Ast == swapinds(Ats, (s, s'), (t, t'))

      swapinds!(Ats, (s, s'), (t, t'))
      @test Ast == Ats
    end
  end #End "ITensor other index operations"

  @testset "Converting Real and Complex Storage" begin
    @testset "Add Real and Complex" for eltype in (Float32, Float64)
      i = Index(2, "i")
      j = Index(2, "j")
      TC = randomITensor(complex(eltype), i, j)
      TR = randomITensor(eltype, i, j)

      S1 = TC + TR
      S2 = TR + TC
      @test typeof(storage(S1)) == NDTensors.Dense{complex(eltype),Vector{complex(eltype)}}
      @test typeof(storage(S2)) == NDTensors.Dense{complex(eltype),Vector{complex(eltype)}}
      for ii in 1:dim(i), jj in 1:dim(j)
        @test S1[i => ii, j => jj] ≈ TC[i => ii, j => jj] + TR[i => ii, j => jj]
        @test S2[i => ii, j => jj] ≈ TC[i => ii, j => jj] + TR[i => ii, j => jj]
      end
    end
  end

  @testset "ITensor, NDTensors.Dense{$SType} storage" for SType in (
    Float32, Float64, ComplexF32, ComplexF64
  )
    mi, mj, mk, ml, mα = 2, 3, 4, 5, 6, 7
    i = Index(mi, "i")
    j = Index(mj, "j")
    k = Index(mk, "k")
    l = Index(ml, "l")
    α = Index(mα, "alpha")

    atol = eps(real(SType)) * 500

    @testset "Set and get values with IndexVals" begin
      A = ITensor(SType, i, j, k)
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        A[k => kk, j => jj, i => ii] = invdigits(SType, ii, jj, kk)
      end
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        @test A[j => jj, k => kk, i => ii] == invdigits(SType, ii, jj, kk)
      end
      @test A[1] == invdigits(SType, 1, 1, 1)
    end
    @testset "Test permute(ITensor,Index...)" begin
      A = randomITensor(SType, i, k, j)
      permA = permute(A, k, j, i)
      @test k == inds(permA)[1]
      @test j == inds(permA)[2]
      @test i == inds(permA)[3]
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        @test A[k => kk, i => ii, j => jj] == permA[i => ii, j => jj, k => kk]
      end
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        @test A[k => kk, i => ii, j => jj] == permA[i => ii, j => jj, k => kk]
      end
      # TODO: I think this was doing slicing, but what is the output
      # of slicing an ITensor?
      #@testset "getindex and setindex with vector of IndexVals" begin
      #    k_inds = [k=>kk for kk ∈ 1:dim(k)]
      #    for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
      #      @test A[k_inds,i=>ii,j=>jj]==permA[i=>ii,j=>jj,k_inds...]
      #    end
      #    for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
      #        A[k_inds,i=>ii,j=>jj]=collect(1:length(k_inds))
      #    end
      #    permA = permute(A,k,j,i)
      #    for ii ∈ 1:dim(i), jj ∈ 1:dim(j)
      #      @test A[k_inds,i=>ii,j=>jj]==permA[i=>ii,j=>jj,k_inds...]
      #    end
      #end
    end
    @testset "Set and get values with Ints" begin
      A = ITensor(SType, i, j, k)
      A = permute(A, k, i, j)
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        A[kk, ii, jj] = invdigits(SType, ii, jj, kk)
      end
      A = permute(A, i, j, k)
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        @test A[ii, jj, kk] == invdigits(SType, ii, jj, kk)
      end
    end
    @testset "Test scalar(::ITensor)" begin
      x = SType(34)
      A = ITensor(x)
      @test x == scalar(A)
      A = ITensor(SType, i, j, k)
      @test_throws DimensionMismatch scalar(A)
    end
    @testset "Test norm(ITensor)" begin
      A = randomITensor(SType, i, j, k)
      @test norm(A) ≈ sqrt(scalar(dag(A) * A))
    end
    @testset "Test dag(::Number)" begin
      x = 1.2 + 2.3im
      @test dag(x) == 1.2 - 2.3im
      x = 1.4
      @test dag(x) == 1.4
    end
    @testset "Test add ITensors" begin
      A = randomITensor(SType, i, j, k)
      B = randomITensor(SType, k, i, j)
      C = A + B
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        @test C[i => ii, j => jj, k => kk] ==
          A[j => jj, i => ii, k => kk] + B[i => ii, k => kk, j => jj]
      end
      @test array(permute(C, i, j, k)) ==
        array(permute(A, i, j, k)) + array(permute(B, i, j, k))
    end

    @testset "Test array" begin
      A = randomITensor(SType, i, j, k)
      B = randomITensor(SType, i, j)
      C = randomITensor(SType, i)

      @test array(permute(A, j, i, k)) == array(A, j, i, k)
      @test_throws DimensionMismatch matrix(A, j, i, k)
      @test_throws DimensionMismatch vector(A, j, i, k)

      @test array(permute(B, j, i)) == array(B, j, i)
      @test matrix(permute(B, j, i)) == matrix(B, j, i)
      @test_throws DimensionMismatch vector(B, j, i)

      @test array(permute(C, i)) == array(C, i)
      @test vector(permute(C, i)) == vector(C, i)
      @test vector(C) == vector(C, i)
      @test_throws DimensionMismatch matrix(C, i)
    end

    @testset "Test factorizations of an ITensor" begin
      A = randomITensor(SType, i, j, k, l)

      @testset "Test SVD of an ITensor" begin
        U, S, V, spec, u, v = svd(A, (j, l))
        @test storage(S) isa NDTensors.Diag{real(SType),Vector{real(SType)}}
        @test A ≈ U * S * V
        @test U * dag(prime(U, u)) ≈ δ(SType, u, u') atol = atol
        @test V * dag(prime(V, v)) ≈ δ(SType, v, v') atol = atol
      end

      @testset "Test SVD of an ITensor with different algorithms" begin
        U, S, V, spec, u, v = svd(A, j, l; alg="recursive")
        @test storage(S) isa NDTensors.Diag{real(SType),Vector{real(SType)}}
        @test A ≈ U * S * V
        @test U * dag(prime(U, u)) ≈ δ(SType, u, u') atol = atol
        @test V * dag(prime(V, v)) ≈ δ(SType, v, v') atol = atol

        U, S, V, spec, u, v = svd(A, j, l; alg="divide_and_conquer")
        @test storage(S) isa NDTensors.Diag{real(SType),Vector{real(SType)}}
        @test A ≈ U * S * V
        @test U * dag(prime(U, u)) ≈ δ(SType, u, u') atol = atol
        @test V * dag(prime(V, v)) ≈ δ(SType, v, v') atol = atol

        U, S, V, spec, u, v = svd(A, j, l; alg="qr_iteration")
        @test storage(S) isa NDTensors.Diag{real(SType),Vector{real(SType)}}
        @test A ≈ U * S * V
        @test U * dag(prime(U, u)) ≈ δ(SType, u, u') atol = atol
        @test V * dag(prime(V, v)) ≈ δ(SType, v, v') atol = atol

        @test_throws ErrorException svd(A, j, l; alg="bad_alg")
      end

      #@testset "Test SVD of a DenseTensor internally" begin
      #  Lis = commoninds(A,IndexSet(j,l))
      #  Ris = uniqueinds(A,Lis)
      #  Lpos,Rpos = NDTensors.getperms(inds(A),Lis,Ris)
      #  # XXX this function isn't used anywhere in ITensors
      #  # (it is no longer needed because of the combiner)
      #  Ut,St,Vt,spec = svd(NDTensors.tensor(A), Lpos, Rpos)
      #  U = itensor(Ut)
      #  S = itensor(St)
      #  V = itensor(Vt)
      #  u = commonind(U, S)
      #  v = commonind(V, S)
      #  @test storage(S) isa NDTensors.Diag{Float64,Vector{Float64}}
      #  @test A≈U*S*V
      #  @test U*dag(prime(U,u))≈δ(SType,u,u') atol = atol
      #  @test V*dag(prime(V,v))≈δ(SType,v,v') atol = atol
      #end

      @testset "Test SVD truncation" begin
        ii = Index(4)
        jj = Index(4)
        T = randomITensor(ComplexF64, ii, jj)
        U, S, V = svd(T, ii; maxdim=2)
        u, s, v = svd(matrix(T))
        @test norm(U * S * V - T) ≈ sqrt(s[3]^2 + s[4]^2)
      end

      @testset "Test QR decomposition of an ITensor" begin
        Q, R = qr(A, (i, l))
        @test eltype(Q) <: eltype(A)
        @test eltype(R) <: eltype(A)
        q = commonind(Q, R)
        @test A ≈ Q * R atol = atol
        @test Q * dag(prime(Q, q)) ≈ δ(SType, q, q') atol = atol

        Q, R = qr(A, (i, j, k, l))
        @test eltype(Q) <: eltype(A)
        @test eltype(R) <: eltype(A)
        q = commonind(Q, R)
        @test hassameinds(Q, (q, i, j, k, l))
        @test hassameinds(R, (q,))
        @test A ≈ Q * R atol = atol
        @test Q * dag(prime(Q, q)) ≈ δ(SType, q, q') atol = atol
      end

      @testset "Regression test for QR decomposition of an ITensor with all indices on one side" begin
        a = Index(2, "a")
        b = Index(2, "b")
        Vab = randomITensor(a, b)
        Q, R = qr(Vab, (a, b))
        @test hasinds(Q, (a, b))
        @test Vab ≈ Q * R atol = atol
      end

      @testset "Test polar decomposition of an ITensor" begin
        U, P, u = polar(A, (k, l))
        @test A ≈ U * P atol = atol
        #Note: this is only satisfied when left dimensions
        #are greater than right dimensions
        UUᵀ = U * dag(prime(U, u))

        # TODO: use a combiner to combine the u indices to make
        # this test simpler
        for ii in 1:dim(u[1]), jj in 1:dim(u[2]), iip in 1:dim(u[1]), jjp in 1:dim(u[2])
          val = UUᵀ[u[1] => ii, u[2] => jj, u[1]' => iip, u[2]' => jjp]
          if ii == iip && jj == jjp
            @test val ≈ one(SType) atol = atol
          else
            @test val ≈ zero(SType) atol = atol
          end
        end
      end

      @testset "Test Hermitian eigendecomposition of an ITensor" begin
        is = IndexSet(i, j)
        T = randomITensor(SType, is..., prime(is)...)
        T = T + swapprime(dag(T), 0, 1)
        D, U, spec, l, r = eigen(T; ishermitian=true)
        @test T ≈ prime(U) * D * dag(U) atol = atol
        UUᴴ = U * prime(dag(U), r)
        @test UUᴴ ≈ δ(r, r')
      end

      @testset "Test factorize of an ITensor" begin
        @testset "factorize default" begin
          L, R = factorize(A, (j, l))
          l = commonind(L, R)
          @test A ≈ L * R
          @test L * dag(prime(L, l)) ≈ δ(SType, l, l')
          @test R * dag(prime(R, l)) ≉ δ(SType, l, l')
        end

        @testset "factorize ortho left" begin
          L, R = factorize(A, (j, l); ortho="left")
          l = commonind(L, R)
          @test A ≈ L * R
          @test L * dag(prime(L, l)) ≈ δ(SType, l, l')
          @test R * dag(prime(R, l)) ≉ δ(SType, l, l')
        end

        @testset "factorize ortho right" begin
          L, R = factorize(A, (j, l); ortho="right")
          l = commonind(L, R)
          @test A ≈ L * R
          @test L * dag(prime(L, l)) ≉ δ(SType, l, l')
          @test R * dag(prime(R, l)) ≈ δ(SType, l, l')
        end

        @testset "factorize ortho none" begin
          L, R = factorize(A, (j, l); ortho="none")
          l = commonind(L, R)
          @test A ≈ L * R
          @test L * dag(prime(L, l)) ≉ δ(SType, l, l')
          @test R * dag(prime(R, l)) ≉ δ(SType, l, l')
        end

        @testset "factorize when ITensor has primed indices" begin
          A = randomITensor(i, i')
          L, R = factorize(A, i)
          l = commonind(L, R)
          @test A ≈ L * R
          @test L * dag(prime(L, l)) ≈ δ(SType, l, l')
          @test R * dag(prime(R, l)) ≉ δ(SType, l, l')

          @test_throws ErrorException factorize(A, i; which_decomp="svd", svd_alg="bad_alg")
        end
      end # End factorize tests

      @testset "Test error for bad decomposition inputs" begin
        @test_throws ErrorException svd(A)
        @test_throws ErrorException factorize(A)
        @test_throws ErrorException eigen(A, inds(A), inds(A))
      end
    end
  end # End Dense storage test

  @testset "dag copy behavior" begin
    i = Index(4, "i")

    v1 = randomITensor(i)
    cv1 = dag(v1)
    cv1[1] = -1
    @test v1[1] ≈ cv1[1]

    v2 = randomITensor(i)
    cv2 = dag(ITensors.NeverAlias(), v2)
    orig_elt = v2[1]
    cv2[1] = -1
    @test v2[1] ≈ orig_elt

    v2 = randomITensor(i)
    cv2 = dag(v2; allow_alias=false)
    orig_elt = v2[1]
    cv2[1] = -1
    @test v2[1] ≈ orig_elt

    v3 = randomITensor(ComplexF64, i)
    orig_elt = v3[1]
    cv3 = dag(v3)
    cv3[1] = -1
    @test v3[1] ≈ orig_elt

    v4 = randomITensor(ComplexF64, i)
    cv4 = dag(ITensors.NeverAlias(), v4)
    orig_elt = v4[1]
    cv4[1] = -1
    @test v4[1] ≈ orig_elt
  end

  @testset "filter ITensor indices" begin
    i = Index(2, "i")
    A = randomITensor(i, i')
    @test hassameinds(filterinds(A; plev=0), (i,))
    @test hassameinds(inds(A; plev=0), (i,))
    is = inds(A)
    @test hassameinds(filterinds(is; plev=0), (i,))
    @test hassameinds(inds(is; plev=0), (i,))
  end

  @testset "product/apply" begin
    s1 = Index(2, "s1")
    s2 = Index(2, "s2")
    s3 = Index(2, "s3")

    rA = Index(3, "rA")
    lA = Index(3, "lA")

    rB = Index(3, "rB")
    lB = Index(3, "lB")

    # operator * operator
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', dag(s1), dag(s2), lB, rB)
    AB = product(A, B)
    @test hassameinds(AB, (s1', s2', s1, s2, lA, rA, lB, rB))
    @test AB ≈ mapprime(prime(A; inds=(s1', s2', s1, s2)) * B, 2 => 1)

    # operator * operator, common dangling indices
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', dag(s1), dag(s2), dag(lA), dag(rA))
    AB = product(A, B)
    @test hassameinds(AB, (s1', s2', s1, s2))
    @test AB ≈ mapprime(prime(A; inds=(s1', s2', s1, s2)) * B, 2 => 1)

    # operator * operator, apply_dag, common dangling indices
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', dag(s1), dag(s2), lB, rB)
    ABAdag = product(A, B; apply_dag=true)
    AB = mapprime(prime(A; inds=(s1', s2', s1, s2)) * B, 2 => 1)
    Adag = swapprime(dag(A), 0 => 1; inds=(s1', s2', s1, s2))
    @test hassameinds(ABAdag, (s1', s2', s1, s2, lB, rB))
    @test ABAdag ≈ mapprime(prime(AB; inds=(s1', s2', s1, s2)) * Adag, 2 => 1)

    # operator * operator, more complicated
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', s3', dag(s1), dag(s2), dag(s3), lB, rB, dag(rA))
    AB = product(A, B)
    @test hassameinds(AB, (s1', s2', s3', s1, s2, s3, lA, lB, rB))
    @test AB ≈ mapprime(prime(A; inds=(s1', s2', s1, s2)) * B, 2 => 1)

    # state * operator (1)
    A = randomITensor(dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', dag(s1), dag(s2), lB, rB)
    AB = product(A, B)
    @test hassameinds(AB, (s1, s2, lA, rA, lB, rB))
    @test AB ≈ mapprime(prime(A; inds=(s1, s2)) * B)

    # state * operator (2)
    A = randomITensor(dag(s1'), dag(s2'), lA, rA)
    B = randomITensor(s1', s2', dag(s1), dag(s2), lB, rB)
    @test_throws ErrorException product(A, B)

    # operator * state (1)
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', lB, rB)
    @test_throws ErrorException product(A, B)

    # operator * state (2)
    A = randomITensor(s1', s2', dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1, s2, lB, rB, dag(lA))
    AB = product(A, B)
    @test hassameinds(AB, (s1, s2, rA, lB, rB))
    @test AB ≈ noprime(A * B)

    # state * state (1)
    A = randomITensor(dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1, s2, lB, rB)
    AB = product(A, B)
    @test hassameinds(AB, (lA, rA, lB, rB))
    @test AB ≈ A * B

    # state * state (2)
    A = randomITensor(dag(s1'), dag(s2'), lA, rA)
    B = randomITensor(s1, s2, lB, dag(rA))
    AB = product(A, B)
    @test hassameinds(AB, (s1', s2', s1, s2, lA, lB))
    @test AB ≈ A * B

    # state * state (3)
    A = randomITensor(dag(s1'), dag(s2'), lA, rA)
    B = randomITensor(s1, s2, lB, rB)
    @test_throws ErrorException product(A, B)

    # state * state (4)
    A = randomITensor(dag(s1), dag(s2), lA, rA)
    B = randomITensor(s1', s2', lB, rB)
    @test_throws ErrorException product(A, B)

    # state * state (5)
    A = randomITensor(dag(s1'), dag(s2'), lA, rA)
    B = randomITensor(s1', s2', lB, rB)
    @test_throws ErrorException product(A, B)
  end

  @testset "inner ($ElType)" for ElType in (Float64, ComplexF64)
    i = Index(2)
    j = Index(2)
    A = randomITensor(ElType, i', j', i, j)
    x = randomITensor(ElType, i, j)
    y = randomITensor(ElType, i, j)
    @test inner(x, y) ≈ (dag(x) * y)[]
    @test inner(x', A, y) ≈ (dag(x)' * A * y)[]
    # No automatic priming
    @test_throws DimensionMismatch inner(x, A, y)
  end

  @testset "hastags" begin
    i = Index(2, "i, x")
    j = Index(2, "j, x")
    A = randomITensor(i, j)
    @test hastags(A, "i")
    @test anyhastags(A, "i")
    @test !allhastags(A, "i")
    @test allhastags(A, "x")
  end

  @testset "directsum" for space in (identity, d -> [QN(0) => d, QN(1) => d]),
    index_op in (identity, dag)

    x = Index(space(2), "x")
    i1 = Index(space(3), "i1")
    j1 = Index(space(4), "j1")
    i2 = Index(space(5), "i2")
    j2 = Index(space(6), "j2")

    A1 = randomITensor(i1, x, j1)
    A2 = randomITensor(x, j2, i2)

    # Generate indices automatically.
    # Reverse the arrow directions in the QN case as a
    # regression test for:
    # https://github.com/ITensor/ITensors.jl/pull/1178.
    S1, s1 = directsum(
      A1 => index_op.((i1, j1)), A2 => index_op.((i2, j2)); tags=["sum_i", "sum_j"]
    )

    # Provide indices
    i1i2 = directsum(i1, i2; tags="sum_i")
    j1j2 = directsum(j1, j2; tags="sum_j")
    s2 = [i1i2, j1j2]
    S2 = directsum(s2, A1 => index_op.((i1, j1)), A2 => index_op.((i2, j2)))
    for (S, s) in zip((S1, S2), (s1, s2))
      for vx in 1:dim(x)
        proj = dag(onehot(x => vx))
        A1_vx = A1 * proj
        A2_vx = A2 * proj
        S_vx = S * proj
        for m in 1:dim(s[1]), n in 1:dim(s[2])
          if m ≤ dim(i1) && n ≤ dim(j1)
            @test S_vx[s[1] => m, s[2] => n] == A1_vx[i1 => m, j1 => n]
          elseif m > dim(i1) && n > dim(j1)
            @test S_vx[s[1] => m, s[2] => n] == A2_vx[i2 => m - dim(i1), j2 => n - dim(j1)]
          else
            @test S_vx[s[1] => m, s[2] => n] == 0
          end
        end
      end
    end

    i1, i2, j, k, l = Index.(space.((2, 3, 4, 5, 6)), ("i1", "i2", "j", "k", "l"))

    A = randomITensor(i1, i2, j)
    B = randomITensor(i1, i2, k)
    C = randomITensor(i1, i2, l)

    S, s = directsum(A => index_op(j), B => index_op(k))
    @test dim(s) == dim(j) + dim(k)
    @test hassameinds(S, (i1, i2, s))

    S, s = (A => index_op(j)) ⊕ (B => index_op(k))
    @test dim(s) == dim(j) + dim(k)
    @test hassameinds(S, (i1, i2, s))

    S, s = directsum(A => index_op(j), B => index_op(k), C => index_op(l))
    @test dim(s) == dim(j) + dim(k) + dim(l)
    @test hassameinds(S, (i1, i2, s))

    @test_throws ErrorException directsum(A => index_op(i2), B => index_op(i2))

    S, (s,) = directsum(A => (index_op(j),), B => (index_op(k),))
    @test s == uniqueind(S, A)
    @test dim(s) == dim(j) + dim(k)
    @test hassameinds(S, (i1, i2, s))

    S, ss = directsum(A => index_op.((i2, j)), B => index_op.((i2, k)))
    @test length(ss) == 2
    @test dim(ss[1]) == dim(i2) + dim(i2)
    @test hassameinds(S, (i1, ss...))

    S, ss = directsum(A => (index_op(j),), B => (index_op(k),), C => (index_op(l),))
    s = only(ss)
    @test s == uniqueind(S, A)
    @test dim(s) == dim(j) + dim(k) + dim(l)
    @test hassameinds(S, (i1, i2, s))

    S, ss = directsum(
      A => index_op.((i2, i1, j)), B => index_op.((i1, i2, k)), C => index_op.((i1, i2, l))
    )
    @test length(ss) == 3
    @test dim(ss[1]) == dim(i2) + dim(i1) + dim(i1)
    @test dim(ss[2]) == dim(i1) + dim(i2) + dim(i2)
    @test dim(ss[3]) == dim(j) + dim(k) + dim(l)
    @test hassameinds(S, ss)
  end

  @testset "ishermitian" begin
    s = Index(2, "s")
    Sz = ITensor([0.5 0.0; 0.0 -0.5], s', s)
    Sp = ITensor([0.0 1.0; 0.0 0.0], s', s)
    @test ishermitian(Sz)
    @test !ishermitian(Sp)
  end

  @testset "convert_eltype, convert_leaf_eltype, $new_eltype" for new_eltype in
                                                                  (Float32, ComplexF64)
    s = Index(2)
    A = randomITensor(s)
    @test eltype(A) == Float64

    Af32 = convert_eltype(new_eltype, A)
    @test Af32 ≈ A
    @test eltype(Af32) == new_eltype

    Af32_2 = convert_leaf_eltype(new_eltype, A)
    @test eltype(Af32_2) == new_eltype
    @test Af32_2 ≈ A

    As1 = [A, A]
    As1_f32 = convert_leaf_eltype(new_eltype, As1)
    @test length(As1_f32) == length(As1)
    @test typeof(As1_f32) == typeof(As1)
    @test eltype(As1_f32[1]) == new_eltype
    @test eltype(As1_f32[2]) == new_eltype

    As2 = [[A, A], [A]]
    As2_f32 = convert_leaf_eltype(new_eltype, As2)
    @test length(As2_f32) == length(As2)
    @test typeof(As2_f32) == typeof(As2)
    @test eltype(As2_f32[1][1]) == new_eltype
    @test eltype(As2_f32[1][2]) == new_eltype
    @test eltype(As2_f32[2][1]) == new_eltype
  end

  @testset "nullspace $eltype" for (ss, sl, sr) in [
      ([QN(-1) => 2, QN(1) => 3], [QN(-1) => 2], [QN(0) => 3]), (5, 2, 3)
    ],
    eltype in (Float32, Float64, ComplexF32, ComplexF64),
    nullspace_kwargs in ((;))
    #nullspace_kwargs in ((; atol=eps(real(eltype)) * 100), (;))

    s, l, r = Index.((ss, sl, sr), ("s", "l", "r"))
    A = randomITensor(eltype, dag(l), s, r)
    N = nullspace(A, dag(l); nullspace_kwargs...)
    @test Base.eltype(N) === eltype
    n = uniqueind(N, A)
    @test op("I", n) ≈ N * dag(prime(N, n))
    @test hassameinds(N, (s, r, n))
    @test norm(A * N) ≈ 0 atol = eps(real(eltype)) * 100
    @test dim(l) + dim(n) == dim((s, r))
    A′, (rn,) = ITensors.directsum(A => (l,), dag(N) => (n,); tags=["⊕"])
    @test dim(rn) == dim((s, r))
    @test norm(A * dag(prime(A, l))) ≈ norm(A * dag(A′))
  end

  @testset "nullspace regression test" begin
    # This is a case that failed before we raised
    # the default atol value in the `nullspace` function
    M = [
      0.663934 0.713867 -0.458164 -1.79885 -0.83443
      1.19064 -1.3474 -0.277555 -0.177408 0.408656
    ]
    i = Index(2)
    j = Index(5)
    A = ITensor(M, i, j)
    N = nullspace(A, i)
    n = uniqueindex(N, A)
    @test dim(n) == dim(j) - dim(i)
  end
end # End Dense ITensor basic functionality

# Disable debug checking once tests are completed
ITensors.disable_debug_checks()

nothing
