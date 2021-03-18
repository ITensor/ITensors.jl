using ITensors
using Test

@testset "Test debug checks on IndexSet construction" begin
  i = Index(2, "i")

  @test !ITensors.using_debug_checks()
  # Test that no error is thrown in constructor
  @test IndexSet(i, i) isa IndexSet
  @test IndexSet(i, i') isa IndexSet

  # Turn on debug checks
  ITensors.enable_debug_checks()
  @test ITensors.using_debug_checks()
  @test_throws ErrorException IndexSet(i, i)
  # Test that no error is thrown in constructor
  @test IndexSet(i, i') isa IndexSet

  # Turn off debug checks
  ITensors.disable_debug_checks()
  @test !ITensors.using_debug_checks()
  # Test that no error is thrown in constructor
  @test IndexSet(i, i) isa IndexSet
  @test IndexSet(i, i') isa IndexSet
end

