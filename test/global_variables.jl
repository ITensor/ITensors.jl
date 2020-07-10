using ITensors
using Test

@testset "Warn ITensor order" begin
  @test get_warn_itensor_order() == ITensors.default_warn_itensor_order
  @test set_warn_itensor_order!(4) == ITensors.default_warn_itensor_order
  @test get_warn_itensor_order() == 4
  @test reset_warn_itensor_order!() == 4
  @test get_warn_itensor_order() == ITensors.default_warn_itensor_order
end

