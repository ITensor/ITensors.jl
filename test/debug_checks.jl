using ITensors
using Test

@testset "Test debug checks on IndexSet construction" begin
  i = Index(2, "i")

  @test !ITensors.use_debug_checks()
  # Test that no error is thrown in constructor
  @test IndexSet(i, i) isa IndexSet{2}
  @test IndexSet(i, i') isa IndexSet{2}

  # Turn on debug checks
  ITensors.enable_debug_checks()
  @test ITensors.use_debug_checks()
  @test_throws ErrorException IndexSet(i, i)
  # Test that no error is thrown in constructor
  @test IndexSet(i, i') isa IndexSet{2}

  # Turn off debug checks
  ITensors.disable_debug_checks()
  @test !ITensors.use_debug_checks()
  # Test that no error is thrown in constructor
  @test IndexSet(i, i) isa IndexSet{2}
  @test IndexSet(i, i') isa IndexSet{2}
end

