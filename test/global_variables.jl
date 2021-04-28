using ITensors
using Test

@testset "Warn ITensor order" begin
  # Check it starts at default value
  @test ITensors.get_warn_order() == ITensors.default_warn_order

  # Set to 4 and reset
  @test ITensors.set_warn_order(4) == ITensors.default_warn_order
  @test ITensors.get_warn_order() == 4
  @test ITensors.reset_warn_order() == 4
  @test ITensors.get_warn_order() == ITensors.default_warn_order

  # Disable it (set to nothing) and reset
  @test ITensors.disable_warn_order() == ITensors.default_warn_order
  @test isnothing(ITensors.get_warn_order())
  @test isnothing(ITensors.reset_warn_order())
  @test ITensors.get_warn_order() == ITensors.default_warn_order

  # Disable macro
  @test ITensors.get_warn_order() == ITensors.default_warn_order
  ITensors.set_warn_order(6)
  @test ITensors.get_warn_order() == 6
  @disable_warn_order begin
    @test isnothing(ITensors.get_warn_order())
  end
  @test ITensors.get_warn_order() == 6
  ITensors.reset_warn_order()
  @test ITensors.get_warn_order() == ITensors.default_warn_order

  # Set macro
  @test ITensors.get_warn_order() == ITensors.default_warn_order
  ITensors.set_warn_order(6)
  @test ITensors.get_warn_order() == 6
  @set_warn_order 10 begin
    @test ITensors.get_warn_order() == 10
  end
  @test ITensors.get_warn_order() == 6
  ITensors.reset_warn_order()
  @test ITensors.get_warn_order() == ITensors.default_warn_order

  # Reset macro
  @test ITensors.get_warn_order() == ITensors.default_warn_order
  ITensors.set_warn_order!(6)
  @test ITensors.get_warn_order() == 6
  @reset_warn_order begin
    @test ITensors.get_warn_order() == ITensors.default_warn_order
  end
  @test ITensors.get_warn_order() == 6
  ITensors.reset_warn_order()
  @test ITensors.get_warn_order() == ITensors.default_warn_order
end
