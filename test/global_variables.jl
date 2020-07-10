using ITensors
using Test

@testset "Warn ITensor order" begin
  # Check it starts at default value
  @test get_warn_itensor_order() == ITensors.default_warn_itensor_order

  # Set to 4 and reset
  @test set_warn_itensor_order!(4) == ITensors.default_warn_itensor_order
  @test get_warn_itensor_order() == 4
  @test reset_warn_itensor_order!() == 4
  @test get_warn_itensor_order() == ITensors.default_warn_itensor_order

  # Disable it (set to nothing) and reset
  @test disable_warn_itensor_order!() == ITensors.default_warn_itensor_order
  @test isnothing(get_warn_itensor_order())
  @test isnothing(reset_warn_itensor_order!())
  @test get_warn_itensor_order() == ITensors.default_warn_itensor_order
end

