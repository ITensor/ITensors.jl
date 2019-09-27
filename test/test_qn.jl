using ITensors,
      Test
import ITensors.SmallString

@testset "QN" begin
  @testset "QNVal Basics" begin
    qv = QNVal(("Sz",0))
    @test name(qv) == SmallString("Sz")
    @test val(qv) == 0
    @test modulus(qv) == 1
  end
end
