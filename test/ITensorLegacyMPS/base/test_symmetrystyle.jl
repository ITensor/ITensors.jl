using ITensors
using ITensors.NDTensors
using Test

@testset "SymmetryStyle trait" begin
  sqn = siteinds("S=1/2", 10; conserve_qns=true)
  s = removeqns(sqn)
  psi = MPS(s)
  psiqn = MPS(sqn)
  @test @inferred(ITensors.SymmetryStyle, ITensors.symmetrystyle(psi)) == ITensors.NonQN()
  @test @inferred(ITensors.SymmetryStyle, ITensors.symmetrystyle(psiqn)) ==
    ITensors.HasQNs()
end
