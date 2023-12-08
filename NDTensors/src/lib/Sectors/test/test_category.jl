import NDTensors.Sectors:
  basename,
  Category,
  @CategoryType_str,
  dynamic,
  DynamicCategory,
  fusion_rule,
  groupdim,
  is_dynamic,
  is_static,
  level,
  static,
  SU,
  Ising,
  Z,
  str_to_val,
  val_to_str
using Test

@testset "Test Category Type" begin
  @testset "Constructors" begin
    C = Category("Ising")
    @test basename(C) == "Ising"

    C = Category("SU", 2)
    @test basename(C) == "SU"
    @test groupdim(C) == 2
    @test level(C) == 0
  end

  @testset "Static and Dynamic Category Types" begin
    D = Category("C")
    @test is_dynamic(D)
    @test !is_static(D)

    S = static(D)
    @test !is_dynamic(S)
    @test is_static(S)

    # Test that calling static twice is ok
    S = static(S)
    @test is_static(S)

    @test basename(S) == basename(D)
    @test groupdim(S) == groupdim(D)
    @test level(S) == level(D)

    D2 = dynamic(S)
    @test is_dynamic(D2)
    @test D2 == D

    # Test that we can call dynamic twice on a Category
    D2 = dynamic(D2)
    @test is_dynamic(D2)
  end

  @testset "Value Dispatch" begin
    # Define a collection of methods overloaded on
    # the basename value
    test_fn(C::CategoryType"A") = "test_fn: got A"
    test_fn(C::CategoryType"B") = "test_fn: got B"
    test_fn(C::DynamicCategory) = test_fn(static(C))

    A1 = Category("A", 1)
    B2 = Category("B", 2)
    C3 = Category("C", 3)

    @test test_fn(A1) == "test_fn: got A"
    @test test_fn(B2) == "test_fn: got B"
    @test_throws MethodError test_fn(C3)
  end

  @testset "Fusion Rule Functions" begin
    @test fusion_rule(Z(2),0,0) == [0]
    @test fusion_rule(Z(2),0,1) == [1]
    @test fusion_rule(Z(2),1,1) == [0]

    @test fusion_rule(Ising,1,1) == [0]
    @test fusion_rule(Ising,1/2,1/2) == [0,1]

    @test fusion_rule(SU(2),1,1) == [0,1,2]
    @test fusion_rule(SU(2),1/2,1) == [1/2,3/2]
  end

  @testset "String to Val and Vice Versa" begin
    @test val_to_str(Ising,0) == "1"
    @test val_to_str(Ising,1/2) == "σ"
    @test val_to_str(Ising,1) == "ψ"

    @test str_to_val(Ising,"1") == 0
    @test str_to_val(Ising,"σ") == 1/2
    @test str_to_val(Ising,"ψ") == 1

    @test_throws ErrorException str_to_val(SU(2),"test")
  end
end

nothing
