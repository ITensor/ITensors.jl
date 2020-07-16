ITensors v0.1.20 Release Notes
==============================
* Make ITensors compatible with Julia v1.3 (#468)
* New function filterinds, alias for inds (#466)
* Add QN ITensor from Array constructor (#464)

ITensors v0.1.19 Release Notes
==============================

* Add setindex!(::MPS, _, ::Colon) (PR #463)
  * Set new limits to limits of input MPS
* Add macros for warn ITensor order (PR #461)
  * Add macros for warn ITensor order
  * Shorten warn ITensor order function name (breaking for anyone who
  managed to use them in the short time they existed).
* Make map for MPS reset the orthogonality limits (PR #460)
  * Makes map and map! reset the orthogonality limits by default.
  * Add keyword argument set_limits to map and map! to let users turn
on and off setting the orthogonality limits (so it can be turned
off for cases like priming).
  * Add orthogonalize, an out-of-place version of orthogonalize!.
  * Add SiteType"S=\1/2" as an alias for SiteType"S=1/2".

ITensors v0.1.18 Release Notes
==============================

* Add functions for controlling warn itensor order (PR #458)
* Add pair syntax to mapprime, replacetags, and replaceinds (PR #459)

ITensors v0.1.17 Release Notes
==============================

* Miscellaneous new ITensor and MPS/MPO functionality (PR #457):
  * Add `eachindex(T::ITensor)` to return an iterator over each cartesian
index of an ITensor (i.e. for an `d x d` ITensor, either `1:d^2` or
`(1,1), (1,2), ..., (d, d)`). For sparse ITensors, this includes
structurally zero and nonzero entries.
  * Add `iterate(A::ITensor, args...)`, which allows using `for a in A
@show a end` to print all elements (zero and nonzero, for sparse
tensors).
  * Add `setindex!(T::ITensor, x::Number, I::CartesianIndex)` to allow
indexing with a `CartesianIndex`, which is naturally returned by
functions like `eachindex`.
  * Add `hasplev(pl::Int)` that returns a function `x -> hasplev(x, pl)`
(useful in functions like `map`).
  * Add `hasind[s](i::Index)` that returns a function `x -> hasind[s](x, i)`
(useful in functions like `map`).
  * Add `hascommoninds(A, B; kwargs...)` which returns true if `A` and `B`
have common indices.
  * Add `findfirstsiteind(M::MPS/MPO, s::Index)` that returns which site
of the MPS/MPO has the site index `s`.
  * Add `findfirstsiteinds(M::MPS/MPO, is)` that returns which site
of the MPS/MPO has the site indices `is`.
  * Add `linkinds(::MPS/MPO)` that returns a vector of the link indices.
  * Add `linkdim(::MPS/MPO, ::Int)` that returns the dimension of the
specified link, and nothing if there is no link found.
  * Add `linkdims(::MPS/MPO)` that returns a vector of the link
dimensions.
  * Fix a bug in `+(::MPST, ::MPST)` that the inputs were getting modified
(the inputs were getting orthogonalized and the prime levels were beging
modified).
  * Add `productMPS(sites, state::Union{String, Int})` to create a uniform
MPS (for example, `productMPS(sites, "Up")` makes a state with all Up
spins).
* Add QR option for factorize (only Dense tensors so far). Used by default
if not truncation is requested (PR #427)

ITensors v0.1.16 Release Notes
==============================

* Update physics site definitions to user newer overload style (PR #453)
* Fix some issues with precompile_itensors.jl code and automatically test it (PR #452)

ITensors v0.1.15 Release Notes
==============================

* Add multi-site op support (PR #444)
* Update state system to be user-extensible using StateName (PR #449)
* Update siteinds system to be more easily extensible using `space` and `siteind` functions (PR #446)
* Remove parenthesis from AutoMPO syntax from tests and examples (PR #448)

ITensors v0.1.14 Release Notes
==============================

* Fix AutoMPO issue #440 (PR #445)
* Have ITensors.compile() compile QN DMRG (PR #442)
* Make linkind return nothing for all links outside the boundary of the MPS (PR #441)

ITensors v0.1.13 Release Notes
==============================

* New ITensors.compile() routine (PR #436, PR #439)
* Propagate keyword args through orthogonalize! (PR #438)
* Speed improvement to op (PR #435)
* Major improvements to op function system (PR #406)

ITensors v0.1.12 Release Notes
==============================

* HDF5 Support for QNITensors, QNIndex (PR #433)
* Add ProjMPO_MPS to exports

ITensors v0.1.11 Release Notes
==============================

* Add tests for contraction bug. Add tests for extended Spectrum definition (PR #432)
* Add ProjMPO_MPS to exports

ITensors v0.1.10 Release Notes
==============================

* Fix missing return statement in QNVal constructor (PR #431)

ITensors v0.1.9 Release Notes
==============================

* Fix bug with AutoMPO dimension in certain cases (PR #426)

ITensors v0.1.8 Release Notes
==============================

* Fix a bug in broadcast and in-place contraction (#425)

ITensors v0.1.7 Release Notes
==============================

* Add Unicode support for SmallStrings/Tags (PR #413)
* Speed up small ITensor contractions (PR #423)
* Add swapsites keyword argument to `replacebond` (PR #420)
* Change `flux(::AbstractMPS)` to return nothing in non-QN case (PR #419)

ITensors v0.1.6 Release Notes
==============================

* Allow user to control Arrow direction of combined Index in combiner (PR #417)
* Fix eigen for case when left/right indices had mixed Arrow directions (PR #417)
* Add exp for QN ITensor (PR #402)
* Add Advanced Usage Guide to docs (PR #387)

ITensors v0.1.5 Release Notes
==============================

* Fix bug with combiner (uncombining step) when combined Index is not the first one (PR #401)
* Add check to ProjMPO to ensure result of `product` is same order as input tensor (PR #390)

ITensors v0.1.4 Release Notes
==============================

* Add note to docs about requiring Julia 1.4 currently
* Improve error message for non-scalar input to `scalar` (PR #396)
* Export @TagType_str macro (PR #393)
* Fix `productMPS` for complex element type (PR #392)

ITensors v0.1.3 Release Notes
==============================

* Use NDTensors v0.1.3, which fixes a bug when taking the SVD of a complex QN ITensor.

ITensors v0.1.2 Release Notes
==============================

* Add functions `norm(::MPS)`, `norm(::MPO)`, `inner(::MPO, ::MPO)`, as well as `logdot`/`loginner` and `lognorm` for getting the logarithm of the inner product or norm between MPSs/MPOs.

