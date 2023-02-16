using ITensors, Test

@testset "README Examples" begin
  @testset "ITensor Basics" begin
    i = Index(3)
    j = Index(5)
    k = Index(2)
    l = Index(7)

    A = ITensor(i, j, k)
    B = ITensor(j, l)

    A[i => 1, j => 1, k => 1] = 11.1
    A[i => 2, j => 1, k => 2] = -21.2
    A[k => 1, i => 3, j => 1] = 31.1  # can provide Index values in any order
    # ...

    # Contract over shared index j
    C = A * B

    @test hasinds(C, i, k, l) == true

    D = randomITensor(k, j, i) # ITensor with random elements

    # Add two ITensors
    # must have same set of indices
    # but can be in any order
    R = A + D
  end

  @testset "SVD of a Matrix" begin
    i = Index(10)
    j = Index(20)
    M = randomITensor(i, j)
    U, S, V = svd(M, i)
    @test norm(M - U * S * V) < 1E-12
  end

  @testset "SVD of a Tensor" begin
    i = Index(4, "i")
    j = Index(4, "j")
    k = Index(4, "k")
    l = Index(4, "l")
    T = randomITensor(i, j, k, l)
    U, S, V = svd(T, i, k)
    @test hasinds(U, i, k)
    @test hasinds(V, j, l)
    @test norm(T - U * S * V) < 1E-12
  end

  @testset "Making Tensor Indices" begin
    i = Index(3)     # Index of dimension 3
    @test dim(i) == 3     # dim(i) = 3

    ci = copy(i)
    @test ci == i    # true

    j = Index(5, "j") # Index with a tag "j"

    @test j != i     # false

    s = Index(2, "n=1,Site") # Index with two tags,
    # "Site" and "n=1"
    @test hastags(s, "Site") # hastags(s,"Site") = true
    @test hastags(s, "n=1")  # hastags(s,"n=1") = true

    i1 = prime(i) # i1 has a "prime level" of 1
    # but otherwise same properties as i
    @test i1 != i # false, prime levels do not match
  end

  @testset "DMRG" begin
    N = 100
    sites = siteinds("S=1", N)

    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network
    ampo = OpSum()
    for j in 1:(N - 1)
      add!(ampo, "Sz", j, "Sz", j + 1)
      add!(ampo, 0.5, "S+", j, "S-", j + 1)
      add!(ampo, 0.5, "S-", j, "S+", j + 1)
    end
    H = MPO(ampo, sites)

    # Create an initial random matrix product state
    psi0 = randomMPS(sites)

    sweeps = Sweeps(2)
    maxdim!(sweeps, 10, 20, 100, 100, 200)
    cutoff!(sweeps, 1E-10)

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    #println("Final energy = $energy")
  end
end

nothing
