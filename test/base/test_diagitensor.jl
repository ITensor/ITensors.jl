using ITensors
using ITensors.NDTensors
using LinearAlgebra
using Test

@testset "diagITensor" begin
  d = 3
  i = Index(d, "i")
  j = Index(d, "j")
  k = Index(d, "k")
  l = Index(d, "l")
  m = Index(d, "m")
  n = Index(d, "n")
  o = Index(d, "o")
  p = Index(d, "p")
  q = Index(d, "q")

  v = collect(1:d)
  vr = randn(d)

  @testset "non-uniform diagonal values" begin
    @testset "diagITensor constructor (no vector, order 2)" begin
      D = diagITensor(i, j)

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d
        if ii == jj
          @test D[i => ii, j => jj] == 0.0
        else
          @test D[i => ii, j => jj] == 0.0
        end
      end
    end

    @testset "diagITensor constructor (no vector, order 3)" begin
      D = diagITensor(i, j, k)

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test D[i => ii, j => jj, k => kk] == 0.0
        else
          @test D[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "diagITensor constructor (no vector, complex)" begin
      D = diagITensor(ComplexF64, i, j)

      @test eltype(D) == ComplexF64
      for ii in 1:d, jj in 1:d
        if ii == jj
          @test D[i => ii, j => jj] == complex(0.0)
        else
          @test D[i => ii, j => jj] == complex(0.0)
        end
      end
    end

    @testset "diag" for ElType in (Float64, ComplexF64)
      A = diagITensor(randn(ElType, d), i, j)
      dA = diag(A)
      @test dA isa DenseTensor{ElType,1}
      @test dA[1] == A[1, 1]
      @test dA[2] == A[2, 2]
    end

    @testset "diagITensor constructor (vector, order 2)" begin
      D = diagITensor(v, i, j)

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d
        if ii == jj
          @test D[i => ii, j => jj] == v[ii]
        else
          @test D[i => ii, j => jj] == 0.0
        end
      end
    end

    @testset "diagITensor constructor (vector, order 3)" begin
      D = diagITensor(v, i, j, k)

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test D[i => ii, j => jj, k => kk] == v[ii]
        else
          @test D[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "diagITensor constructor (complex)" begin
      vc = v + im * v
      D = diagITensor(vc, i, j, k)

      @test eltype(D) == ComplexF64
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test D[i => ii, j => jj, k => kk] == vc[ii]
        else
          @test D[i => ii, j => jj, k => kk] == complex(0.0)
        end
      end
    end

    @testset "Complex operations" begin
      xr = randn(d)
      xi = randn(d)
      D = diagITensor(xr + im * xi, i, j, k)
      @test eltype(D) == ComplexF64
      rD = real(D)
      iD = imag(D)
      @test eltype(rD) == Float64
      @test eltype(iD) == Float64
      @test typeof(storage(rD)) <: NDTensors.Diag
      @test norm(rD + im * iD - D) < 1E-8
    end

    @testset "Constructor AllowAlias/NeverAlias" begin
      vv = ones(d)
      D = diagITensor(vv, i, j)
      @test eltype(D) === Float64
      D[1, 1] = 5.0
      @test vv[1] == 1.0
      @test vv[1] != D[1, 1]

      vv = ones(Int, d)
      D = diagITensor(vv, i, j)
      @test eltype(D) === Float64
      D[1, 1] = 5.0
      @test vv[1] == 1.0
      @test vv[1] != D[1, 1]

      vv = ones(Int, d)
      D = diagITensor(Int, vv, i, j)
      @test eltype(D) === Int
      D[1, 1] = 5
      @test vv[1] == 1
      @test vv[1] != D[1, 1]

      vv = ones(d)
      D = diagitensor(vv, i, j)
      @test eltype(D) === Float64
      D[1, 1] = 5.0
      @test vv[1] == 5.0
      @test vv[1] == D[1, 1]

      vv = ones(Int, d)
      D = diagitensor(vv, i, j)
      @test eltype(D) === Float64
      D[1, 1] = 5.0
      @test vv[1] == 1.0
      @test vv[1] != D[1, 1]

      vv = ones(Int, d)
      D = diagitensor(Int, vv, i, j)
      @test eltype(D) === Int
      D[1, 1] = 5
      @test vv[1] == 5
      @test vv[1] == D[1, 1]

      D = diagITensor(1, i, j)
      @test eltype(D) === Float64
      D[1, 1] = 5
      @test D[1, 1] == 5

      D = diagITensor(Int, 1, i, j)
      @test eltype(D) === Int
      D[1, 1] = 5
      @test D[1, 1] == 5
    end

    @testset "fill!" begin
      D = diagITensor(ones(d), i, j, k)
      D = fill!(D, 2.0)
      for ii in 1:d
        @test D[i => ii, j => ii, k => ii] == 2.0
      end

      @test eltype(D) == Float64
    end

    @testset "Set elements" begin
      D = diagITensor(i, j, k)

      for ii in 1:d
        D[i => ii, j => ii, k => ii] = ii
      end

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test D[i => ii, j => jj, k => kk] == ii
        else
          @test D[i => ii, j => jj, k => kk] == 0.0
        end
      end

      # Can't set off-diagonal elements
      @test_throws ErrorException D[i => 2, j => 1, k => 1] = 0.0
      @test_throws ErrorException D[i => 1, j => 2, k => 1] = 0.0
    end

    @testset "Convert diag to dense" begin
      D = diagITensor(v, i, j, k)
      T = dense(D)

      @test storage(T) isa NDTensors.Dense{Float64}
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test T[ii, ii, ii] == ii
        else
          @test T[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "Convert diag to dense with denseblocks" begin
      D = diagITensor(v, i, j, k)
      T = denseblocks(D)

      @test storage(T) isa NDTensors.Dense{Float64}
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test T[ii, ii, ii] == ii
        else
          @test T[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "Add (Diag + Diag)" begin
      v1 = randn(d)
      v2 = randn(d)
      D1 = diagITensor(v1, i, j, k)
      D2 = diagITensor(v2, k, i, j)

      v3 = v1 + v2
      D3 = D1 + D2

      @test D3 ≈ dense(D1) + dense(D2)
      for ii in 1:d
        @test D3[ii, ii, ii] == v3[ii]
      end
    end

    @testset "Add ( number * Diag + Diag)" begin
      v1 = randn(d)
      v2 = randn(d)
      D1 = Float32(2.0) * diagITensor(v1, i, j, k)
      D2 = diagITensor(v2, k, i, j)

      v3 = 2 * v1 + v2
      D3 = D1 + D2

      @test D3 ≈ dense(D1) + dense(D2)
      for ii in 1:d
        @test D3[ii, ii, ii] == v3[ii]
      end
    end

    @testset "Add (Diag uniform + Diag uniform)" begin
      D1 = δ(i, j, k)
      D2 = δ(k, i, j)

      D3 = D1 + D2

      @test D3 ≈ dense(D1) + dense(D2)
    end

    @testset "Add (Diag + Dense)" begin
      D = diagITensor(vr, i, j, k)
      A = randomITensor(k, j, i)

      R = D + A

      @test R ≈ dense(D) + A
      for ii in 1:d
        @test R[ii, ii, ii] ≈ D[ii, ii, ii] + A[ii, ii, ii]
      end
    end

    @testset "Add (Dense + Diag)" begin
      D = diagITensor(vr, i, j, k)
      A = randomITensor(i, k, j)

      R = A + D

      @test R ≈ dense(D) + A
      for ii in 1:d
        @test R[ii, ii, ii] ≈ D[ii, ii, ii] + A[ii, ii, ii]
      end
    end

    @testset "Contraction (all contracted)" begin
      D = diagITensor(v, i, j, k)
      A = randomITensor(j, k, i)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (all contracted) with different types" begin
      D = diagITensor(v, i, j, k)
      A = randomITensor(Float32, j, k, i)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A

      D = diagITensor(v, i, j, k)
      A = randomITensor(ComplexF32, j, k, i)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (all dense contracted)" begin
      D = diagITensor(v, j, k, i)
      A = randomITensor(i, j)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction Diag*Dense (general)" begin
      D = diagITensor(v, l, i, k, j)
      A = randomITensor(m, k, n, l)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction Diag*Dense (outer)" begin
      D = diagITensor(v, l, i, k, j)
      A = randomITensor(m, n)

      @test order(D * A) == 6
      @test D * A ≈ dense(D) * A
    end

    @testset "Contraction Diag*Diag (outer)" begin
      D1 = diagITensor(v, l, i)
      D2 = diagITensor(v, m, n)

      @test order(D1 * D2) == 4
      @test D1 * D2 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction Diag*Diag (all contracted)" begin
      D1 = diagITensor(v, l, i, k, j)
      D2 = diagITensor(vr, j, l, i, k)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction Diag*Diag (general)" begin
      D1 = diagITensor(v, l, i, k, j)
      D2 = diagITensor(vr, m, k, n, l)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction Diag*Diag (no contracted)" begin
      D1 = diagITensor(v, i, j)
      D2 = diagITensor(vr, k, l)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction Diag*Scalar" begin
      D = diagITensor(v, i, j)
      x = 2.0

      @test x * D ≈ x * dense(D)
      @test D * x ≈ x * dense(D)

      xc = 2 + 3im

      @test xc * D ≈ xc * dense(D)
      @test D * xc ≈ xc * dense(D)
    end
  end

  @testset "Uniform diagonal ITensor" begin
    @testset "delta constructor (order 2)" begin
      D = δ(i, j)

      @test eltype(D) == Float64
      for ii in 1:d, jj in 1:d
        if ii == jj
          @test D[i => ii, j => jj] == 1.0
        else
          @test D[i => ii, j => jj] == 0.0
        end
      end
    end

    @testset "delta constructor (order 3)" begin
      D = δ(i, j, k)

      @test eltype(D) == Float64
      for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k)
        if ii == jj == kk
          @test D[i => ii, j => jj, k => kk] == 1.0
        else
          @test D[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "Set elements" begin
      D = δ(i, j, k)

      @test eltype(D) == Float64

      # Can't set elements of uniform diag tensor
      # TODO: should we make a function that converts
      # to a version that can?
      @test_throws ErrorException D[i => 1, j => 1, k => 1] = 2.0
      @test_throws ErrorException D[i => 2, j => 1, k => 1] = 4.3
      @test_throws ErrorException D[i => 1, j => 2, k => 1] = 2
    end

    @testset "Convert diag uniform to dense" begin
      D = δ(i, j, k)
      T = dense(D)

      @test storage(T) isa NDTensors.Dense{Float64}
      for ii in 1:d, jj in 1:d, kk in 1:d
        if ii == jj == kk
          @test T[ii, ii, ii] == 1.0
        else
          @test T[i => ii, j => jj, k => kk] == 0.0
        end
      end
    end

    @testset "Add (Diag uniform + Dense)" begin
      D = δ(i, j, k)
      A = randomITensor(k, j, i)

      R = D + A

      @test R ≈ dense(D) + A
      for ii in 1:d
        @test R[ii, ii, ii] ≈ D[ii, ii, ii] + A[ii, ii, ii]
      end
    end

    @testset "Contraction (Diag uniform * Dense, all contracted)" begin
      D = δ(i, j, k)
      A = randomITensor(j, k, i)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (Diag uniform * Dense, all dense contracted)" begin
      D = δ(j, k, i)
      A = randomITensor(i, j)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (Diag uniform * Dense, general)" begin
      D = δ(l, i, k, j)
      A = randomITensor(m, k, n, l)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction with different bond dimensions" begin
      idim = 3
      mdim = 2

      i = Index(idim, "i")
      m = Index(mdim, "m")

      A = randomITensor(i, i', m)
      D = δ(i, i')

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (Diag uniform * Dense, replace index)" begin
      D = δ(i, k)
      A = randomITensor(m, k, n, l)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (Diag uniform * Dense, replace index 2)" begin
      D = δ(k, i)
      A = randomITensor(m, n, k, l)

      @test D * A ≈ dense(D) * A
      @test A * D ≈ dense(D) * A
    end

    @testset "Contraction (Diag uniform * Diag uniform, all contracted)" begin
      D1 = δ(l, i, k, j)
      D2 = δ(j, l, i, k)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction (Diag uniform * Diag uniform, general)" begin
      D1 = δ(l, i, k, j)
      D2 = δ(m, k, n, l)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Contraction (Diag uniform * Diag uniform, no contracted)" begin
      D1 = δ(i, j)
      D2 = δ(k, l)

      @test D1 * D2 ≈ dense(D1) * dense(D2)
      @test D2 * D1 ≈ dense(D1) * dense(D2)
    end

    @testset "Rectangular Diag * Dense regression test (#969)" begin
      i = Index(3)
      j = Index(2)
      A = randomITensor(i)
      B = delta(i, j)
      C = A * B
      @test hassameinds(C, j)
      for n in 1:dim(j)
        @test C[n] == A[n]
      end
    end
  end
end

nothing
