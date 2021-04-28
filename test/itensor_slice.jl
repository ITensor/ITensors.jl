using ITensors
using Test
import Random: seed!

seed!(12345)

@testset "Dense ITensor slicing functionality" begin
  i = Index(2)
  j = Index(3)
  k = Index(4)
  l = Index(5)

  A₀ = randomITensor(i, j, k, l)
  a = randn(dim(l), dim(k))

  A = copy(A₀)
  A[l => 1:dim(l), i => 1, k => 1:dim(k), j => 2] = a

  for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
    if ii == 1 && jj == 2
      @test A[j => 2, l => ll, i => 1, k => kk] == a[ll, kk]
    else
      @test A[j => jj, l => ll, i => ii, k => kk] == A₀[j => jj, l => ll, i => ii, k => kk]
    end
  end

  A = copy(A₀)
  A[1, 2, :, :] = transpose(a)

  for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
    if ii == 1 && jj == 2
      @test A[j => 2, l => ll, i => 1, k => kk] == a[ll, kk]
    else
      @test A[j => jj, l => ll, i => ii, k => kk] == A₀[j => jj, l => ll, i => ii, k => kk]
    end
  end

  A = copy(A₀)
  A[l => 1:(dim(l) - 1), i => 1, k => 1:(dim(k) - 1), j => 2] = a[1:(end - 1), 1:(end - 1)]

  for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
    if ii == 1 && jj == 2 && kk ∈ 1:(dim(k) - 1) && ll ∈ 1:(dim(l) - 1)
      @test A[j => 2, l => ll, i => 1, k => kk] == a[ll, kk]
    else
      @test A[j => jj, l => ll, i => ii, k => kk] == A₀[j => jj, l => ll, i => ii, k => kk]
    end
  end

  A = copy(A₀)
  A[k => :, i => 1, l => :, j => 2] = a'

  for ii in 1:dim(i), jj in 1:dim(j), kk in 1:dim(k), ll in 1:dim(l)
    if ii == 1 && jj == 2
      @test A[j => 2, l => ll, i => 1, k => kk] == a[ll, kk]
    else
      @test A[j => jj, l => ll, i => ii, k => kk] == A₀[j => jj, l => ll, i => ii, k => kk]
    end
  end
end
