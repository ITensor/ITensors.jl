using NDTensors.SmallVectors
using Test: @inferred, @test, @testset, @test_broken

using NDTensors.SmallVectors:
  setindex,
  resize,
  push,
  pushfirst,
  pop,
  popfirst,
  append,
  prepend,
  insert,
  deleteat,
  circshift,
  insertsorted,
  insertsorted!,
  insertsortedunique,
  insertsortedunique!,
  mergesorted,
  mergesorted!,
  mergesortedunique,
  mergesortedunique!

function test_smallvectors()
  return @testset "SmallVectors" begin
    x = SmallVector{10}([1, 3, 5])
    mx = MSmallVector(x)

    @test x isa SmallVector{10,Int}
    @test mx isa MSmallVector{10,Int}
    @test eltype(x) === Int
    @test eltype(mx) === Int

    # TODO: Test construction has zero allocations.
    # TODO: Extend construction to arbitrary collections, like tuple.

    # conversion
    @test @inferred(SmallVector(x)) == x
    @test @allocated(SmallVector(x)) == 0
    @test @inferred(SmallVector(mx)) == x
    @test @allocated(SmallVector(mx)) == 0

    # length
    @test @inferred(length(x)) == 3
    @test @allocated(length(x)) == 0
    @test @inferred(length(SmallVectors.buffer(x))) == 10
    @test @allocated(length(SmallVectors.buffer(x))) == 0

    nalloc_limit = 128

    item = 115
    no_broken = (false, false, false, false)
    for (
      f!,
      f,
      ans,
      args,
      nalloc,
      f!_impl_broken,
      f!_noalloc_broken,
      f_impl_broken,
      f_noalloc_broken,
    ) in [
      (:push!, :push, [1, 3, 5, item], (item,), nalloc_limit, no_broken...),
      (:append!, :append, [1, 3, 5, item], ([item],), nalloc_limit, no_broken...),
      (:prepend!, :prepend, [item, 1, 3, 5], ([item],), nalloc_limit, no_broken...),
      (:pushfirst!, :pushfirst, [item, 1, 3, 5], (item,), nalloc_limit, no_broken...),
      (:setindex!, :setindex, [1, item, 5], (item, 2), nalloc_limit, no_broken...),
      (:pop!, :pop, [1, 3], (), nalloc_limit, no_broken...),
      (:popfirst!, :popfirst, [3, 5], (), nalloc_limit, no_broken...),
      (:insert!, :insert, [1, item, 3, 5], (2, item), nalloc_limit, no_broken...),
      (:deleteat!, :deleteat, [1, 5], (2,), nalloc_limit, no_broken...),
      (:circshift!, :circshift, [5, 1, 3], (1,), nalloc_limit, no_broken...),
      (:sort!, :sort, [1, 3, 5], (), nalloc_limit, no_broken...),
      (:insertsorted!, :insertsorted, [1, 2, 3, 5], (2,), nalloc_limit, no_broken...),
      (:insertsorted!, :insertsorted, [1, 3, 3, 5], (3,), nalloc_limit, no_broken...),
      (
        :insertsortedunique!,
        :insertsortedunique,
        [1, 2, 3, 5],
        (2,),
        nalloc_limit,
        no_broken...,
      ),
      (
        :insertsortedunique!,
        :insertsortedunique,
        [1, 3, 5],
        (3,),
        nalloc_limit,
        no_broken...,
      ),
      (:mergesorted!, :mergesorted, [1, 2, 3, 3, 5], ([2, 3],), nalloc_limit, no_broken...),
      (
        :mergesortedunique!,
        :mergesortedunique,
        [1, 2, 3, 5],
        ([2, 3],),
        nalloc_limit,
        no_broken...,
      ),
    ]
      mx_tmp = copy(mx)
      @eval begin
        if VERSION < v"1.7"
          # broken kwarg wasn't added to @test yet
          if $f!_impl_broken
            @test_broken @inferred($f!(copy($mx), $args...)) == $ans
          else
            @test @inferred($f!(copy($mx), $args...)) == $ans
          end
          if $f!_noalloc_broken
            @test_broken @allocated($f!($mx_tmp, $args...)) ≤ $nalloc
          else
            @test @allocated($f!($mx_tmp, $args...)) ≤ $nalloc
          end
          if $f_impl_broken
            @test_broken @inferred($f($x, $args...)) == $ans
          else
            @test @inferred($f($x, $args...)) == $ans
          end
          if $f_noalloc_broken
            @test_broken @allocated($f($x, $args...)) ≤ $nalloc
          else
            @test @allocated($f($x, $args...)) ≤ $nalloc
          end
        else
          @test @inferred($f!(copy($mx), $args...)) == $ans broken = $f!_impl_broken
          @test @allocated($f!($mx_tmp, $args...)) ≤ $nalloc broken = $f!_noalloc_broken
          @test @inferred($f($x, $args...)) == $ans broken = $f_impl_broken
          @test @allocated($f($x, $args...)) ≤ $nalloc broken = $f_noalloc_broken
        end
      end
    end

    # Separated out since for some reason it breaks the `@inferred`
    # check when `kwargs` are interpolated into `@eval`.
    ans, kwargs = [5, 3, 1], (; rev=true)
    mx_tmp = copy(mx)
    @test @inferred(sort!(copy(mx); kwargs...)) == ans
    @test @allocated(sort!(mx_tmp; kwargs...)) == 0
    @test @inferred(sort(x; kwargs...)) == ans
    @test @allocated(sort(x; kwargs...)) ≤ nalloc_limit

    ans, args = [1, 3, 5, item], ([item],)
    @test @inferred(vcat(x, args...)) == ans
    @test @allocated(vcat(x, args...)) ≤ nalloc_limit
  end
end

# TODO: switch to:
# @testset "SmallVectors" test_smallvectors()
# (new in Julia 1.9)
test_smallvectors()
