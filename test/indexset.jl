using ITensors
using Test
using Combinatorics
using Compat

@testset "IndexSet" begin
  idim = 2
  jdim = 3
  kdim = 4
  ldim = 5
  i = Index(idim, "i")
  j = Index(jdim, "j")
  k = Index(kdim, "k")
  l = Index(ldim, "l")
  @testset "show" begin
    indset = IndexSet(i, j, k)
    @test length(sprint(show, indset)) > 1
  end
  @testset "Basic constructors" begin
    I = IndexSet(i, j, k)
    @test IndexSet(I) === I
    @test l ∈ IndexSet(I..., l)
    @test l ∈ IndexSet(l, I...)
    @test length(IndexSet(i, j)) == 2
    # construct with function
    ind_list = [i, j, k]
    I = IndexSet(ii -> ind_list[ii], 3)
    @test i ∈ I
    @test j ∈ I
    @test k ∈ I
    I = IndexSet(ii -> ind_list[ii], Order(3))
    @test i ∈ I
    @test j ∈ I
    @test k ∈ I
  end
  @testset "length of IndexSet and friends" begin
    @test length(IndexSet(i, j)) == 2
    @test size(IndexSet(i, j)) == (length(IndexSet(i, j)),)
  end
  @testset "Convert to Index" begin
    @test Index(IndexSet(i)) === i
    @test_throws BoundsError Index(IndexSet(i, j))
  end
  @testset "Index dimensions" begin
    I = IndexSet(i, j, k)
    @test dim(I) == idim * jdim * kdim
    @test dims(I) == [idim, jdim, kdim]
    @test dim(I, 1) == idim
    @test dim(I, 2) == jdim
    @test dim(I, 3) == kdim

    @test maxdim(I) == max(idim, jdim, kdim)
  end

  @testset "Set operations" begin
    I1 = @inferred(IndexSet(i, j, k))
    I2 = @inferred(IndexSet(k, l))
    I3 = @inferred(IndexSet(j, l))
    @test I1 isa Vector{Index{Int}}
    @test @inferred(hassameinds(I1, (k, j, i)))
    @test @inferred(Nothing, getfirst(setdiff(I1, I2, I3))) == i
    @test isnothing(@inferred(Nothing, getfirst(setdiff(I1, IndexSet(k, j, i)))))
    @test @inferred(setdiff(I1, I2)) == [i, j]
    @test hassameinds(@inferred(setdiff(I1, I2)), IndexSet(i, j))
    @test hassameinds(@inferred(setdiff(I1, I2)), (j, i))
    @test I1 ∩ I2 == [k]
    @test hassameinds(I1 ∩ I2, IndexSet(k))
    @test @inferred(Nothing, getfirst(intersect(I1, I2))) == k
    @test isnothing(@inferred(Nothing, getfirst(intersect(I1, IndexSet(l)))))
    @test @inferred(intersect(I1, IndexSet(j, l))) == [j]
    @test hassameinds(@inferred(intersect(I1, IndexSet(j, l))), IndexSet(j))
    @test @inferred(Nothing, getfirst(intersect(I1, IndexSet(j, l)))) == j
    @test @inferred(intersect(I1, IndexSet(j, k))) == [j, k]
    @test hassameinds(@inferred(intersect(I1, (j, k))), IndexSet(j, k))
    @test hassameinds(@inferred(intersect(I1, (j, k, l))), (j, k))
    @test @inferred(filterinds(I1, "i")) == IndexSet(i)
    @test @inferred(filterinds(I1; tags="i")) == IndexSet(i)
    @test @inferred(filterinds(I1; inds=j)) == IndexSet(j)
    @test @inferred(filterinds(I1; tags="i", inds=j)) == IndexSet()
    @test @inferred(filterinds(I1; plev=1, inds=j)) == IndexSet()
    @test @inferred(filterinds(I1; plev=0, inds=k)) == IndexSet(k)
    @test @inferred(filterinds(I1; plev=0)) == IndexSet(i, j, k)
    @test @inferred(filterinds(I1; inds=l)) == IndexSet()
    @test @inferred(hassameinds(filter(I1, "i"), IndexSet(i)))
    @test @inferred(Nothing, getfirst(I1, "j")) == j
    @test isnothing(@inferred(Nothing, getfirst(I1, "l")))
    @test @inferred(Nothing, findfirst(I1, i)) == 1
    @test @inferred(Nothing, findfirst(I1, j)) == 2
    @test @inferred(Nothing, findfirst(I1, k)) == 3
    @test isnothing(@inferred(Nothing, findfirst(I1, Index(2))))
  end

  @testset "Set operations with Order" begin
    i, j, k, l = Index.(2, ("i", "j", "k", "l"))

    Iij = IndexSet(i, j)
    Ijl = IndexSet(j, l)
    Ikl = IndexSet(k, l)
    Iijk = IndexSet(i, j, k)

    #
    # setdiff 
    # intersect
    # symdiff
    # union
    # filter
    #

    #
    # setdiff
    #

    @test @inferred(setdiff(Iijk, Ikl)) == [i, j]

    @test @inferred(setdiff(Iij, Iijk)) == Index{Int}[]

    @test @inferred(uniqueinds(Iijk, Ikl; tags="i")) == [i]

    @test @inferred(uniqueinds(Iijk, Ikl; tags=not("i"))) == [j]

    @test @inferred(setdiff(Iijk, Ijl, Ikl)) == [i]

    #
    # intersect
    #

    @test @inferred(intersect(Iijk, Ikl)) == [k]

    @test @inferred(intersect(Iijk, Iij)) == [i, j]

    @test @inferred(commoninds(Iijk, Iij; tags="i")) == [i]

    #
    # symdiff
    #

    @test @inferred(symdiff(Iijk, Ikl)) == [i, j, l]

    @test @inferred(symdiff(Iijk, Iij)) == [k]

    #
    # union
    #

    @test @inferred(union(Iijk, Ikl)) == [i, j, k, l]

    @test @inferred(union(Iijk, Iij)) == [i, j, k]
  end

  @testset "intersect index ordering" begin
    I = IndexSet(i, k, j)
    J = IndexSet(j, l, i)
    # Test that intersect respects the ordering
    # of the indices in the first IndexSet
    @test @inferred(hassameinds(intersect(I, J), IndexSet(i, j)))
    @test @inferred(hassameinds(intersect(J, I), IndexSet(j, i)))
  end
  @testset "adjoint" begin
    I = IndexSet(i, k, j)
    @test adjoint(I) == IndexSet(i', k', j')
  end
  @testset "mapprime" begin
    I = IndexSet(i', k'', j)
    @test mapprime(I, 1, 5) == IndexSet(i^5, k'', j)
    @test mapprime(I, 2, 0) == IndexSet(i', k, j)

    J = IndexSet(i, j, k')
    @test mapprime(J, 0, 2) == IndexSet(i'', j'', k')

    J = mapprime(J, 1, 5)
    @test J == IndexSet(i, j, k^5)
  end
  @testset "strides" begin
    I = IndexSet(i, j)
    @test NDTensors.dim_to_strides(I) == (1, idim)
    @test NDTensors.dim_to_stride(I, 1) == 1
    @test NDTensors.dim_to_stride(I, 2) == idim
  end
  @testset "setprime" begin
    I = IndexSet(i, j)
    J = setprime(I, 2, i)
    @test i'' ∈ J
  end
  @testset "prime" begin
    I = IndexSet(i, j)
    J = prime(I, j)
    @test i ∈ J
    @test j' ∈ J
    J = prime(I; inds=j)
    @test i ∈ J
    @test j' ∈ J
    J = prime(I; inds=not(j))
    @test i' ∈ J
    @test j ∈ J
  end
  @testset "noprime" begin
    I = IndexSet(i'', j')
    J = noprime(I)
    @test i ∈ J
    @test j ∈ J
  end
  @testset "swapprime" begin
    I = IndexSet(i, j)
    @test swapprime(I, 0, 1) == IndexSet(i', j')
    @test swapprime(I, 0, 4) == IndexSet(i^4, j^4)
    I = IndexSet(i, j'')
    @test swapprime(I, 2, 0) == IndexSet(i'', j)
    I = IndexSet(i, j'', k, l)
    @test swapprime(I, 2, 0) == IndexSet(i'', j, k'', l'')
    I = IndexSet(i, k'', j'')
    @test swapprime(I, 2, 1) == IndexSet(i, k', j')
    # In-place version:
    I = IndexSet(i, k'', j''')
    I = swapprime(I, 2, 0)
    @test I == IndexSet(i'', k, j''')
    # With tags specified:
    I = IndexSet(i, k, j)
    @test swapprime(I, 0, 1, "i") == IndexSet(i', k, j)
    @test swapprime(I, 0, 1, "j") == IndexSet(i, k, j')

    I = IndexSet(i, i', j)
    @test swapprime(I, 0, 1, "i") == IndexSet(i', i, j)
    @test swapprime(I, 0, 1, "j") == IndexSet(i, i', j')
  end

  @testset "swaptags" begin
    i1 = Index(2, "Site,A")
    i2 = Index(2, "Site,B")
    is = IndexSet(i1, i2)
    sis = swaptags(is, "Site", "Link")
    for j in sis
      @test !hastags(j, "Site")
      @test hastags(j, "Link")
    end
  end

  @testset "hastags" begin
    i = Index(2, "i, x")
    j = Index(2, "j, x")
    is = IndexSet(i, j)
    @test hastags(is, "i")
    @test anyhastags(is, "i")
    @test !allhastags(is, "i")
    @test allhastags(is, "x")
  end

  @testset "broadcasting" begin
    x = Index([QN(n) => 1 for n in 0:1], "x")
    y = Index([QN(n) => 2 for n in 0:1], "y")
    I = IndexSet(x, y)

    # prime
    J = prime.(I)
    # broken for now
    #@inferred broadcast(prime, I)
    @test J isa IndexSet
    @test x' ∈ J
    @test y' ∈ J

    # prime 2
    J = prime.(I, 2)
    # broken for now
    #@inferred broadcast(prime, I, 2)
    @test J isa IndexSet
    @test x'' ∈ J
    @test y'' ∈ J

    # tag
    J = addtags.(I, "t")
    # broken for now
    #@inferred broadcast(addtags, I, "t")
    @test J isa IndexSet
    @test addtags(x, "t") ∈ J
    @test addtags(y, "t") ∈ J

    # dag
    J = dag.(I)
    # broken for now
    #@inferred broadcast(dag, I)
    @test J isa IndexSet
    @test x ∈ J
    @test y ∈ J
    @test dir(J[1]) == -dir(I[1])
    @test dir(J, x) == -dir(I, x)
    @test dir(J[2]) == -dir(I[2])
    @test dir(J, y) == -dir(I, y)
    @test ITensors.dirs(J, (x, y)) == [-dir(I, x), -dir(I, y)]
    @test ITensors.dirs(J) == [-dir(I, x), -dir(I, y)]

    # dir
    dirsI = dir.(I)
    # broken for now
    #@inferred broadcast(dir, I)
    @test dirsI isa Vector{ITensors.Arrow}
    @test dirsI == [ITensors.Out, ITensors.Out]

    # dims
    dimsI = dim.(I)
    # broken for now
    #@inferred broadcast(dim, I)
    @test dimsI isa Vector{Int}
    @test dimsI == [2, 4]

    # pairs
    J = prime.(I)
    pairsI = I .=> J
    #@inferred broadcast(=>, I, J)
    @test pairsI isa Vector{<:Pair}
    @test pairsI == [x => x', y => y']

    pairsI = I .=> 1
    #@inferred broadcast(=>, I, 1)
    @test pairsI isa Vector{<:Pair}
    @test pairsI == [x => 1, y => 1]

    pairsI = I .=> (1, 2)
    #@inferred broadcast(=>, I, (1, 2))
    @test pairsI isa Vector{<:Pair}
    @test pairsI == [x => 1, y => 2]

    pairsI = I .=> [1, 2]
    #@inferred broadcast(=>, I, [1, 2])
    @test pairsI isa Vector{<:Pair}
    @test pairsI == [x => 1, y => 2]
  end

  @testset "ITensors.indpairs" begin
    si = [QN(0) => 1, QN(1) => 2, QN(2) => 3]
    sj = [QN(0) => 2, QN(1) => 3, QN(2) => 4]
    sk = [QN(0) => 3, QN(1) => 4, QN(2) => 5]
    sl = [QN(0) => 2]
    i, j, k, l = Index.((si, sj, sk, sl), ("i", "j", "k", "l"))
    T = randomITensor(dag(j), k', i', dag(k), j', dag(i))
    ip = ITensors.indpairs(T)
    i1 = first.(ip)
    i2 = last.(ip)
    @test i1' == i2
    for x in i1
      @test dir(x) == dir(T, x)
    end
    for x in i2
      @test dir(x) == dir(T, x)
    end
  end

  @testset "permute" begin
    i, j, k = Index.(Ref([QN() => 2]), ("i", "j", "k"))
    is1 = (dag(i), j, dag(k))
    is2 = (i, dag(j), k)
    for x1 in permutations(is1), x2 in permutations(is2)
      # permute x1 into the ordering of x2
      px1 = permute(x1, x2)
      @test px1 == x2
      for y in x1
        @test dir(x1, y) == dir(px1, y)
        @test -dir(x2, y) == dir(px1, y)
      end
      # permute x2 into the ordering of x1
      px2 = permute(x2, x1)
      @test px2 == x1
      for y in x2
        @test dir(x2, y) == dir(px2, y)
        @test -dir(x1, y) == dir(px2, y)
      end
    end
  end

  @testset "dag" begin
    is = [Index(2), Index(3)]
    @test is == dag(is)
    is = Index[]
    @test dag(is) == Index[]
  end
end

nothing
