using ITensors
using Test

@testset "Test debug checks on IndexSet construction" begin
  i = Index(2, "i")

  initially_using_debug_checks = ITensors.using_debug_checks()

  ITensors.disable_debug_checks()
  @test !ITensors.using_debug_checks()
  # Test that no error is thrown in constructor
  @test ITensor(i, i) isa ITensor
  @test ITensor(i, i') isa ITensor

  # Turn on debug checks
  ITensors.enable_debug_checks()
  @test ITensors.using_debug_checks()
  @test_throws ErrorException ITensor(i, i)
  # Test that no error is thrown in constructor
  @test ITensor(i, i') isa ITensor

  # Turn off debug checks
  ITensors.disable_debug_checks()
  @test !ITensors.using_debug_checks()
  # Test that no error is thrown in constructor
  @test ITensor(i, i) isa ITensor
  @test ITensor(i, i') isa ITensor

  # Reset to the initial value
  if !initially_using_debug_checks
    ITensors.disable_debug_checks()
  else
    ITensors.enable_debug_checks()
  end
  @test ITensors.using_debug_checks() == initially_using_debug_checks
end
