using ITensors,
      LinearAlgebra, Test

a = [-0.1, -0.12]
@test ITensors.truncate!(a) == (0., 0.)
@test length(a) == 1
a = [0.1, 0.01, 1e-13]
@test ITensors.truncate!(a,
                         use_absolute_cutoff=true,
                         cutoff=1e-5) == (1e-13, (0.01 + 1e-13)/2)
@test length(a) == 2

i = Index(2,"i")
j = Index(2,"j")
A = randomITensor(i,j)
@test_throws ErrorException factorize(A, i, dir="left")
@test_throws ErrorException factorize(A, i, ortho="fakedir")

A = randomITensor(i,i')
eigA = eigen(A)
Ut, Dt = eigen(tensor(A))
eigArr = eigen(array(A))
@test diag(array(eigA.D), 0) == eigArr.values
@test diag(array(Dt), 0) == eigArr.values

At = rand(10, 10)
k  = Index(10, "k")
A = itensor(At + transpose(At), k, k')
@test Array(exphermitian(tensor(A))) ≈ exp(At + transpose(At))

@testset "Spectrum" begin
  i = Index(100,"i")
  j = Index(100,"j")

  U,S,V = svd(rand(100,100))
  S ./= norm(S)
  A = itensor(U*ITensors.LinearAlgebra.diagm(0=>S)*V', i,j)

  spec = svd(A,i).spec

  @test eigs(spec) ≈ S .^2
  @test truncerror(spec) == 0.0

  spec = svd(A,i; maxdim=length(S)-3).spec
  @test truncerror(spec) ≈ sum(S[end-2:end].^2)

  @test entropy(Spectrum([0.5; 0.5], 0.0)) == log(2)
  @test entropy(Spectrum([1.0], 0.0)) == 0.0 
  @test entropy(Spectrum([0.0], 0.0)) == 0.0 
end
