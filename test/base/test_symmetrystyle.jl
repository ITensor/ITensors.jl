using ITensors
using ITensors.NDTensors
using Test

@testset "SymmetryStyle trait" begin
  i = Index(2)
  iqn = Index([QN(0) => 1, QN(1) => 2])

  @test @inferred(ITensors.symmetrystyle(i)) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle((i,))) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle([i])) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle(i', i)) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle((i', i))) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle([i', i])) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle(i'', i', i)) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle((i'', i', i))) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle([i'', i', i])) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle(i''', i'', i', i)) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle((i''', i'', i', i))) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle([i''', i'', i', i])) == ITensors.NonQN()

  @test @inferred(ITensors.symmetrystyle(iqn)) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle((iqn,))) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle([iqn])) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle(iqn', iqn)) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle((iqn', iqn))) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle([iqn', iqn])) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle(iqn'', iqn', iqn)) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle((iqn'', iqn', iqn))) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle([iqn'', iqn', iqn])) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle(iqn', i)) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle(i', i, iqn)) == ITensors.HasQNs()
  @test @inferred(ITensors.symmetrystyle((i', i, iqn))) == ITensors.HasQNs()
  @test @inferred(ITensors.SymmetryStyle, ITensors.symmetrystyle([i', i, iqn])) ==
    ITensors.HasQNs()

  A = randomITensor(i', dag(i))
  Aqn = randomITensor(iqn', dag(iqn))

  @test @inferred(ITensors.SymmetryStyle, ITensors.symmetrystyle(A)) == ITensors.NonQN()
  @test @inferred(ITensors.SymmetryStyle, ITensors.symmetrystyle(Aqn)) == ITensors.HasQNs()

  T = Tensor(A)
  Tqn = Tensor(Aqn)

  @test @inferred(ITensors.symmetrystyle(T)) == ITensors.NonQN()
  @test @inferred(ITensors.symmetrystyle(Tqn)) == ITensors.HasQNs()
end
