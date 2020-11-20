using ITensors
using Test

@testset "Warn ITensor order" begin
  # Check it starts at default value
  @test get_warn_order() == ITensors.default_warn_order

  # Set to 4 and reset
  @test set_warn_order!(4) == ITensors.default_warn_order
  @test get_warn_order() == 4
  @test reset_warn_order!() == 4
  @test get_warn_order() == ITensors.default_warn_order

  # Disable it (set to nothing) and reset
  @test disable_warn_order!() == ITensors.default_warn_order
  @test isnothing(get_warn_order())
  @test isnothing(reset_warn_order!())
  @test get_warn_order() == ITensors.default_warn_order

  # Disable macro
  @test get_warn_order() == ITensors.default_warn_order
  set_warn_order!(6)
  @test get_warn_order() == 6
  @disable_warn_order begin
    @test isnothing(get_warn_order())
  end
  @test get_warn_order() == 6
  reset_warn_order!()
  @test get_warn_order() == ITensors.default_warn_order

  # Set macro
  @test get_warn_order() == ITensors.default_warn_order
  set_warn_order!(6)
  @test get_warn_order() == 6
  @set_warn_order 10 begin
    @test get_warn_order() == 10
  end
  @test get_warn_order() == 6
  reset_warn_order!()
  @test get_warn_order() == ITensors.default_warn_order

  # Reset macro
  @test get_warn_order() == ITensors.default_warn_order
  set_warn_order!(6)
  @test get_warn_order() == 6
  @reset_warn_order begin
    @test get_warn_order() == ITensors.default_warn_order
  end
  @test get_warn_order() == 6
  reset_warn_order!()
  @test get_warn_order() == ITensors.default_warn_order
end

