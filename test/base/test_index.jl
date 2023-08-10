using ITensors
using ITensors.NDTensors
using Random
using Test
import ITensors: In, Out, Neither

@testset "Index" begin
  @testset "Index with dim" begin
    i = Index(2)
    @test id(i) != 0
    @test hasid(i, id(i))
    @test dim(i) == 2
    @test dir(i) == Neither
    @test plev(i) == 0
    @test tags(i) == TagSet("")
    @test Int(i) == 2
    @test length(i) == 1
    @test Tuple(i) == (i,)
    @test collect(i)[] === i
  end
  @testset "Index with all args" begin
    i = Index(1, 2, In, "Link", 1)
    @test id(i) == 1
    @test dim(i) == 2
    @test dir(i) == In
    @test plev(i) == 1
    @test tags(i) == TagSet("Link")
    j = copy(i)
    @test id(j) == 1
    @test dim(j) == 2
    @test dir(j) == In
    @test plev(j) == 1
    @test tags(j) == TagSet("Link")
    @test j == i
  end
  @testset "prime" begin
    i = Index(2)
    @test plev(i) == 0
    i2 = prime(i, 2)
    @test plev(i2) == 2
    i1 = i'
    @test plev(i1) == 1
    i2 = i''
    @test plev(i2) == 2
    i3 = i'''
    @test plev(i3) == 3
    i6 = i^6
    @test plev(i6) == 6
    i0 = noprime(i)
    @test plev(i0) == 0
  end
  @testset "IndexVal" begin
    i = Index(2)
    @test_deprecated i[1]
    @test_deprecated i(1)
    @test val(i => 1) == 1
    @test ind(i => 1) == i
    @test isindequal(i, i => 2)
    @test isindequal(i => 2, i)
    @test plev(i' => 2) == 1
    @test val(i' => 2) == 2
    @test plev(prime(i => 2, 4)) == 4

    @test plev(i => 2) == 0
    @test plev(i' => 2) == 1
    @test prime(i => 2) == (i' => 2)
    @test IndexVal(i, 1) == Pair(i, 1)
    iv = i => 2
    ĩv = sim(i => 2)
    @test ind(iv) ≠ ind(ĩv)
    @test val(iv) == val(ĩv)
  end
  @testset "Iteration" begin
    i = Index(3)

    c = 1
    for iv in eachindval(i)
      @test iv == (i => c)
      c += 1
    end

    c = 1
    for n in eachval(i)
      @test n == c
      c += 1
    end
  end
  @testset "Broadcasting" begin
    N = 3
    i = Index(2)
    ps = (n - 1 for n in 1:4)
    is = prime.(i, ps)
    @test is[1] == i
    @test is[2] == i'
    @test is[3] == i''
    ts = ("i$n" for n in 1:4)
    is = settags.(i, ts)
    @test is[1] == addtags(i, "i1")
    @test is[2] == addtags(i, "i2")
    @test is[3] == addtags(i, "i3")
  end
  @testset "Index ID random seed" begin
    Random.seed!(index_id_rng(), 1234)
    i = Index(2)
    j = Index(2)
    Random.seed!(index_id_rng(), 1234)
    ĩ = Index(2)
    j̃ = Index(2)
    Random.seed!(index_id_rng(), 123)
    ĩ′ = Index(2)
    j̃′ = Index(2)
    @test id(i) == id(ĩ)
    @test id(j) == id(j̃)
    @test id(i) ≠ id(ĩ′)
    @test id(j) ≠ id(j̃′)

    Random.seed!(index_id_rng(), 1234)
    Random.seed!(1234)
    i = Index(2)
    j = Index(2)
    Random.seed!(1234)
    ĩ = Index(2)
    j̃ = Index(2)
    Random.seed!(123)
    ĩ′ = Index(2)
    j̃′ = Index(2)
    @test id(i) ≠ id(ĩ)
    @test id(j) ≠ id(j̃)
    @test id(i) ≠ id(ĩ′)
    @test id(j) ≠ id(j̃′)

    Random.seed!(index_id_rng(), 1234)
    Random.seed!(1234)
    i = Index(2)
    j = Index(2)
    A = randomITensor(i, j)
    Random.seed!(1234)
    ĩ = Index(2)
    j̃ = Index(2)
    Ã = randomITensor(ĩ, j̃)
    Random.seed!(123)
    ĩ′ = Index(2)
    j̃′ = Index(2)
    Ã′ = randomITensor(ĩ′, j̃′)
    @test id(i) ≠ id(ĩ)
    @test id(j) ≠ id(j̃)
    @test id(i) ≠ id(ĩ′)
    @test id(j) ≠ id(j̃′)
    @test all(tensor(A) .== tensor(Ã))
    @test all(tensor(A) .≠ tensor(Ã′))

    Random.seed!(index_id_rng(), 1234)
    Random.seed!(1234)
    i = Index(2)
    j = Index(2)
    A = randomITensor(i, j)
    Random.seed!(index_id_rng(), 1234)
    Random.seed!(1234)
    ĩ = Index(2)
    j̃ = Index(2)
    Ã = randomITensor(ĩ, j̃)
    Random.seed!(index_id_rng(), 1234)
    Random.seed!(123)
    ĩ′ = Index(2)
    j̃′ = Index(2)
    Ã′ = randomITensor(ĩ′, j̃′)
    @test id(i) == id(ĩ)
    @test id(j) == id(j̃)
    @test id(i) == id(ĩ′)
    @test id(j) == id(j̃′)
    @test all(tensor(A) .== tensor(Ã))
    @test all(tensor(A) .≠ tensor(Ã′))
  end
  @testset "directsum" begin
    i = Index(2, "i")
    j = Index(3, "j")
    ij = directsum(i, j; tags="test")
    @test dim(ij) == dim(i) + dim(j)
    @test hastags(ij, "test")
    k = Index(4, "k")
    ijk = directsum(i, j, k; tags="test2")
    @test dim(ijk) == dim(i) + dim(j) + dim(k)
    @test hastags(ijk, "test2")
  end
end

nothing
