@eval module $(gensym())
using Test: @test, @testset
using NDTensors.NamedDimsArrays:
  NamedDimsArrays,
  NamedDimsArray,
  align,
  dimnames,
  isnamed,
  named,
  namedaxes,
  namedsize,
  unname
using NDTensors: NDTensors
using GPUArraysCore: @allowscalar
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list

@testset "NamedDimsArrays $(@__FILE__)" begin
  @testset "Basic functionality on $dev" for dev in devices_list(ARGS)
    a = dev(randn(3, 4))
    na = named(a, ("i", "j"))
    # TODO: Just call this `size`?
    i, j = namedsize(na)
    # TODO: Just call `namedaxes`?
    ai, aj = namedaxes(na)
    @test !isnamed(a)
    @test isnamed(na)
    @test dimnames(na) == ("i", "j")
    @allowscalar begin
      @test na[1, 1] == a[1, 1]
      na[1, 1] = 11
      @test na[1, 1] == 11
    end
    # TODO: Should `size` output `namedsize`?
    @test size(na) == (3, 4)
    @test namedsize(na) == (named(3, "i"), named(4, "j"))
    @test length(na) == 12
    # TODO: Should `axes` output `namedaxes`?
    @test axes(na) == (1:3, 1:4)
    @test namedaxes(na) == (named(1:3, "i"), named(1:4, "j"))
    @test dev(randn(named(3, "i"), named(4, "j"))) isa NamedDimsArray
    @allowscalar begin
      @test na["i" => 1, "j" => 2] == a[1, 2]
      @test na["j" => 2, "i" => 1] == a[1, 2]
      na["j" => 2, "i" => 1] = 12
      @test na[1, 2] == 12
      @test na[j => 1, i => 2] == a[2, 1]
      @test na[aj => 1, ai => 2] == a[2, 1]
      na[j => 1, i => 2] = 21
      @test na[2, 1] == 21
      na[aj => 1, ai => 2] = 2211
      @test na[2, 1] == 2211
    end
    na′ = align(na, ("j", "i"))
    @test a == permutedims(unname(na′), (2, 1))
    na′ = align(na, (j, i))
    @test a == permutedims(unname(na′), (2, 1))
    na′ = align(na, (aj, ai))
    @test a == permutedims(unname(na′), (2, 1))
  end

  @testset "Shorthand constructors (eltype=$elt)" for dev in devices_list(ARGS),
    elt in (Float32, ComplexF32, Float64, ComplexF64)

    i, j = named.((2, 2), ("i", "j"))
    value = rand(elt)
    for na in (dev(zeros(elt, i, j)), dev(zeros(elt, (i, j))))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar @test iszero(na)
    end
    for na in (dev(fill(value, i, j)), dev(fill(value, (i, j))))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar @test all(isequal(value), na)
    end
    for na in (dev(rand(elt, i, j)), dev(rand(elt, (i, j))))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar begin
        @test !iszero(na)
        @test all(x -> real(x) > 0, na)
      end
    end
    for na in (dev(randn(elt, i, j)), dev(randn(elt, (i, j))))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar @test !iszero(na)
    end
  end

  @testset "Shorthand constructors (eltype=unspecified)" for dev in devices_list(ARGS)
    i, j = named.((2, 2), ("i", "j"))
    default_elt = Float64
    for na in (dev(zeros(i, j)), dev(zeros((i, j))))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar @test iszero(na)
    end
    for na in (dev(rand(i, j)), dev(rand((i, j))))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar begin
        @test !iszero(na)
        @test all(x -> real(x) > 0, na)
      end
    end
    for na in (dev(randn(i, j)), dev(randn((i, j))))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @allowscalar @test !iszero(na)
    end
  end
end
end
