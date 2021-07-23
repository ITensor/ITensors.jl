# Upgrade guide

## Upgrading from ITensors.jl 0.1 to 0.2

The main breaking changes in ITensor.jl v0.2 involve changes to the `ITensor`, `IndexSet`, and `IndexVal` types. Most user code should be fine, but see below for more details.

In addition, we have moved development of NDTensors.jl into ITensors.jl to simplify the development process until NDTensors is more stable and can be a standalone package. Again, see below for more details.

For a more comprehensive list of changes, see the [change log](https://github.com/ITensor/ITensors.jl/blob/main/NEWS.md) and the [commit history on Github](https://github.com/ITensor/ITensors.jl/commits/main).

If you have issues upgrading, please reach out by [raising an issue on Github](https://github.com/ITensor/ITensors.jl/issues/new) or asking a question on the [ITensor support forum](http://itensor.org/support/).

Also make sure to run your code with `julia --depwarn=yes` to see warnings about function names and interfaces
that have been deprecated and will be removed in v0.3 of ITensors.jl (these are not listed here).

## Major design changes: changes to the `ITensor`, `IndexSet`, and `IndexVal` types

### Changes to the ITensor type

#### Removal of tensor order type parameter

The tensor order type paramater has been removed from the `ITensor` type, so you can no longer write `ITensor{3}` to specify an order 3 ITensor ([PR #591](https://github.com/ITensor/ITensors.jl/pull/591)). Code that uses the ITensor order type parameter will now lead to the following error:
```julia
julia> i = Index(2)
(dim=2|id=588)

julia> ITensor{2}(i', i)
ERROR: TypeError: in Type{...} expression, expected UnionAll, got Type{ITensor}
Stacktrace:
 [1] top-level scope
   @ REPL[27]:1
```
Simply remove the type parameter:
```julia
julia> ITensor(i', i)
ITensor ord=2 (dim=2|id=913)' (dim=2|id=913)
ITensors.NDTensors.EmptyStorage{ITensors.NDTensors.EmptyNumber, ITensors.NDTensors.Dense{ITensors.NDTensors.EmptyNumber, Vector{ITensors.NDTensors.EmptyNumber}}}
```
Pro tip: from the command line, you can replace all examples like that with:
```bash
find . -type f -iname "*.jl" -exec sed -i 's/ITensor{.*}/ITensor/g' "{}" +
```
Of course, make sure to back up your code before running this!

Additionally, a common code pattern may be using the type parameter for dispatch:
```julia
using ITensors

function mynorm(A::ITensor{N}) where {N}
  return norm(A)^N
end

function mynorm(A::ITensor{1})
  return norm(A)
end

function mynorm(A::ITensor{2})
  return norm(A)^2
end
```
  Instead, you can use an if-statement:
```julia
function mynormN(A::ITensor)
  return norm(A)^order(A)
end

function mynorm1(A::ITensor)
  return norm(A)
end

function mynorm2(A::ITensor)
  return norm(A)^2
end

function mynorm(A::ITensor) 
  return if order(A) == 1
    mynorm1(A)
  elseif order(A) == 2
    mynorm2(A)
  else
    return mynormN(A)
  end
end
```
Alternatively, you can use the `Order` type to dispatch on the
ITensor order as follows:
```julia
function mynorm(::Order{N}, A::ITensor) where {N}
  return norm(A)^N
end

function mynorm(::Order{1}, A::ITensor)
  return norm(A)
end

function mynorm(::Order{2}, A::ITensor)
  return norm(A)^2
end

function mynorm(A::ITensor)
  return mynorm(Order(A), A)
end
```
`Order(A::ITensor)` returns the order of the ITensor (like `order(A::ITensor)`), however
as a type that can be dispatched on. Note that it is not type stable, so there will
be a small runtime overhead for doing this.

#### Change to storage type of Index collection in ITensor

ITensors now store a `Tuple` of `Index` instead of an `IndexSet` ([PR #626](https://github.com/ITensor/ITensors.jl/pull/626)). Therefore, calling `inds` on
an ITensor will now just return a `Tuple`:
```julia
julia> i = Index(2)
(dim=2|id=770)

julia> j = Index(3)
(dim=3|id=272)

julia> A = randomITensor(i, j)
ITensor ord=2 (dim=2|id=770) (dim=3|id=272)
ITensors.NDTensors.Dense{Float64, Vector{Float64}}

julia> inds(A)
((dim=2|id=770), (dim=3|id=272))
```
while before it returned an `IndexSet` (in fact, the `IndexSet` type has been removed, see below for details). In general, this should not affect user code, since a `Tuple` of `Index` should have all of the same functions defined for it that `IndexSet` did. If you find this is not the case, please [raise an issue on Github](https://github.com/ITensor/ITensors.jl/issues/new) or on the [ITensor support forum](http://itensor.org/support/).

#### ITensor type now directly wraps a Tensor

The ITensor type no longer has separate field `inds` and `store`, just a single field `tensor` (PR #626). In general you should not be accessing the fields directly, instead you should be using the functions `inds(A::ITensor)` and `storage(A::ITensor)`, so this should not affect most code. However, in case you have code like:
```julia
i = Index(2)
A = randomITensor(i)
A.inds
```
this will error in v0.2 with:
```julia
julia> A.inds
ERROR: type ITensor has no field inds
Stacktrace:
 [1] getproperty(x::ITensor, f::Symbol)
   @ Base ./Base.jl:33
 [2] top-level scope
   @ REPL[43]:1
```
and you should change it to:
```julia
inds(A)
```

### Changes to the ITensor constructors

#### Plain ITensor constructors now return ITensors with `EmptyStorage` storage

`ITensor` constructors from collections of `Index`, such as `ITensor(i, j, k)`, now return an `ITensor` with `EmptyStorage` (previously called `Empty`) storage instead of `Dense` or `BlockSparse` storage filled with 0 values. Most operations should still work that worked previously, but please contact us if there are issues ([PR #641](https://github.com/ITensor/ITensors.jl/pull/641)).

For example:
```julia
julia> i = Index(2)
(dim=2|id=346)

julia> A = ITensor(i', dag(i))
ITensor ord=2 (dim=2|id=346)' (dim=2|id=346)
ITensors.NDTensors.EmptyStorage{ITensors.NDTensors.EmptyNumber, ITensors.NDTensors.Dense{ITensors.NDTensors.EmptyNumber, Vector{ITensors.NDTensors.EmptyNumber}}}

julia> A' * A
ITensor ord=2 (dim=2|id=346)'' (dim=2|id=346)
ITensors.NDTensors.EmptyStorage{ITensors.NDTensors.EmptyNumber, ITensors.NDTensors.Dense{ITensors.NDTensors.EmptyNumber, Vector{ITensors.NDTensors.EmptyNumber}}}
```
so now contracting two `EmptyStorage` ITensors returns another `EmptyStorage` ITensor. You can allocate the storage by setting elements of the ITensor:
```julia
julia> A[i' => 1, i => 1] = 0.0
0.0

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=346)'
Dim 2: (dim=2|id=346)
ITensors.NDTensors.Dense{Float64, Vector{Float64}}
 2×2
 0.0  0.0
 0.0  0.0
```
Additionally, it will take on the element type of the first value set:
```julia
julia> A = ITensor(i', dag(i))
ITensor ord=2 (dim=2|id=346)' (dim=2|id=346)
ITensors.NDTensors.EmptyStorage{ITensors.NDTensors.EmptyNumber, ITensors.NDTensors.Dense{ITensors.NDTensors.EmptyNumber, Vector{ITensors.NDTensors.EmptyNumber}}}

julia> A[i' => 1, i => 1] = 1.0 + 0.0im
1.0 + 0.0im

julia> @show A;
A = ITensor ord=2
Dim 1: (dim=2|id=346)'
Dim 2: (dim=2|id=346)
ITensors.NDTensors.Dense{ComplexF64, Vector{ComplexF64}}
 2×2
 1.0 + 0.0im  0.0 + 0.0im
 0.0 + 0.0im  0.0 + 0.0im
```
If you have issues upgrading, please let us know.

#### Slight change to automatic conversion of element type when constructing ITensor from Array

`ITensor` constructors from `Array` now only convert to floating point for `Array{Int}` and `Array{Complex{Int}}`. That same conversion is added for QN ITensor constructors to be consistent with non-QN versions ([PR #620](https://github.com/ITensor/ITensors.jl/pull/620)). Previously it tried to convert arrays of any element type to the closest floating point type with Julia's `float` function. This should not affect most user code.

### Changes to the IndexSet type

The `IndexSet` type has been removed in favor of Julia's `Tuple` and `Vector` types ([PR #626](https://github.com/ITensor/ITensors.jl/pull/626)). `ITensor`s now contain a `Tuple` of `Index`, while set operations like `commoninds` that used to return `IndexSet` now return a `Vector` of `Index`:
```julia
julia> i = Index(2)
(dim=2|id=320)

julia> A = randomITensor(i', i)
ITensor ord=2 (dim=2|id=320)' (dim=2|id=320)
ITensors.NDTensors.Dense{Float64, Vector{Float64}}

julia> inds(A) # Previously returned IndexSet, now returns Tuple
((dim=2|id=320)', (dim=2|id=320))

julia> commoninds(A', A) # Previously returned IndexSet, now returns Vector
1-element Vector{Index{Int64}}:
 (dim=2|id=320)'
```

To help with upgrading code, `IndexSet{IndexT}` has been redefined as a type alias for `Vector{IndexT<:Index}` (which is subject to change to some other collection of indices, and likely will be removed in ITensors v0.3). Therefore it no longer has a type parameter for the number of indices, similar to the change to the `ITensor` type. If you were using the plain `IndexSet` type, code should generally still work properly. However, if you were using the type parameters of `IndexSet`, such as:
```julia
function myorder2(is::IndexSet{N}) where {N}
  return N^2
end
```
then you will need to remove the type parameter and rewrite your code generically to accept `Tuple` or `Vector`, such
as:
```julia
function myorder2(is)
  return length(is)^2
end
```
In general you should be able to just remove usages of `IndexSet` in your code, and can just use `Tuple` or `Vector` of `Index` instead, such as change `is = IndexSet(i, j, k)` to `is = (i, j, k)` or `is = [i, j, k]`. Priming, tagging, and set operations now work generically on those types. If you see issues with upgrading your code, please let us know.

### Changes to the IndexVal type

Similar to the removal of `IndexSet`, we have also removed the `IndexVal` type ([PR #665](https://github.com/ITensor/ITensors.jl/pull/665)). Now, all use cases of `IndexVal` can be replaced by using Julia's `Pair` type, for example instead of:
```julia
i = Index(2)
IndexVal(i, 2)
```
use:
```julia
i = Index(2)
i => 2
# Or:
Pair(i, 2)
```
Note that we have made `IndexVal{IndexT}` an alias for `Pair{IndexT,Int}`, so code using `IndexVal` such as `IndexVal(i, 2)` should generally still work. However, we encourage users to change from `IndexVal(i, 2)` to `i => 2`.

## NDTensors.jl package now being developed internally within ITensors.jl

The `NDTensors` module has been moved into the `ITensors` package, so `ITensors` no longer depends on the standalone `NDTensors` package. This should only effect users who were using both `NDTensors` and `ITensors` seperately. If you want to use the latest `NDTensors` library, you should do `using ITensors.NDTensors` instead of `using NDTensors`, and will need to install `ITensors` with `using Pkg; Pkg.add("ITensors")` in order to use the latest versions of `NDTensors`. Note the current `NDTensors.jl` package will still exist, but for now developmentof `NDTensors` will occur within `ITensors.jl` ([PR #650](https://github.com/ITensor/ITensors.jl/pull/650)).

## Miscellaneous breaking changes

### `state` function renamed `val`, `state` given a new more general definition

Rename the `state` functions currently defined for various site types to `val` for mapping a string name for an index to an index value (used in ITensor indexing and MPS construction). `state` functions now return single-index ITensors representing various single-site states ([PR #664](https://github.com/ITensor/ITensors.jl/pull/664)). So now to get an Index value from a string, you use:
```julia
N = 10
s = siteinds("S=1/2", N)
val(s[1], "Up") == 1
val(s[1], "Dn") == 2
```
`state` now returns an ITensor corresponding to the state with that value as the only nonzero element:
```julia
julia> @show state(s[1], "Up");
state(s[1], "Up") = ITensor ord=1
Dim 1: (dim=2|id=597|"S=1/2,Site,n=1")
ITensors.NDTensors.Dense{Float64, Vector{Float64}}
 2-element
 1.0
 0.0

julia> @show state(s[1], "Dn");
state(s[1], "Dn") = ITensor ord=1
Dim 1: (dim=2|id=597|"S=1/2,Site,n=1")
ITensors.NDTensors.Dense{Float64, Vector{Float64}}
 2-element
 0.0
 1.0
```
which allows for more general states to be defined, such as:
```julia
julia> @show state(s[1], "X+");
state(s[1], "X+") = ITensor ord=1
Dim 1: (dim=2|id=597|"S=1/2,Site,n=1")
ITensors.NDTensors.Dense{Float64, Vector{Float64}}
 2-element
 0.7071067811865475
 0.7071067811865475

julia> @show state(s[1], "X-");
state(s[1], "X-") = ITensor ord=1
Dim 1: (dim=2|id=597|"S=1/2,Site,n=1")
ITensors.NDTensors.Dense{Float64, Vector{Float64}}
 2-element
  0.7071067811865475
 -0.7071067811865475
```
which will be used for making more general MPS product states.

This should not affect end users in general, besides ones who had customized the previous `state` function, such as with overloads like:
```julia
ITensors.state(::SiteType"My_S=1/2", ::StateName"Up") = 1
ITensors.state(::SiteType"My_S=1/2", ::StateName"Dn") = 2
```
which should be changed now to:
```julia
ITensors.val(::SiteType"My_S=1/2", ::StateName"Up") = 1
ITensors.val(::SiteType"My_S=1/2", ::StateName"Dn") = 2
```

### `"Qubit"` site type QN convention change

The QN convention of the `"Qubit"` site type is changed to track the total number of 1 bits instead of the net number of 1 bits vs 0 bits (i.e. change the QN from +1/-1 to 0/1) ([PR #676](https://github.com/ITensor/ITensors.jl/pull/676)).
```julia
julia> s = siteinds("Qubit", 4; conserve_number=true)
4-element Vector{Index{Vector{Pair{QN, Int64}}}}:
 (dim=2|id=925|"Qubit,Site,n=1") <Out>
 1: QN("Number",0) => 1
 2: QN("Number",1) => 1
 (dim=2|id=799|"Qubit,Site,n=2") <Out>
 1: QN("Number",0) => 1
 2: QN("Number",1) => 1
 (dim=2|id=8|"Qubit,Site,n=3") <Out>
 1: QN("Number",0) => 1
 2: QN("Number",1) => 1
 (dim=2|id=385|"Qubit,Site,n=4") <Out>
 1: QN("Number",0) => 1
 2: QN("Number",1) => 1
```
Before it was +1/-1 like `"S=1/2"`:
```julia
julia> s = siteinds("S=1/2", 4; conserve_sz=true)
4-element Vector{Index{Vector{Pair{QN, Int64}}}}:
 (dim=2|id=364|"S=1/2,Site,n=1") <Out>
 1: QN("Sz",1) => 1
 2: QN("Sz",-1) => 1
 (dim=2|id=823|"S=1/2,Site,n=2") <Out>
 1: QN("Sz",1) => 1
 2: QN("Sz",-1) => 1
 (dim=2|id=295|"S=1/2,Site,n=3") <Out>
 1: QN("Sz",1) => 1
 2: QN("Sz",-1) => 1
 (dim=2|id=810|"S=1/2,Site,n=4") <Out>
 1: QN("Sz",1) => 1
 2: QN("Sz",-1) => 1
```
This shouldn't affect end users in general. The new convention is a bit more intuitive since the quantum number
can be thought of as counting the total number of 1 bits in the state, though the conventions can be mapped
to each other with a constant.

### `maxlinkdim` for MPS/MPO with no indices

`maxlinkdim(::MPS/MPO)` returns a minimum of `1` (previously it returned 0 for MPS/MPO without and link indices) ([PR #663](https://github.com/ITensor/ITensors.jl/pull/663)).

