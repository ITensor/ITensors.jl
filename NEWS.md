This file is a (mostly) comprehensive list of changes made in each release of ITensors.jl. For a completely comprehensive but more verbose list, see the [commit history on Github](https://github.com/ITensor/ITensors.jl/commits/main).

While we are in v0.x of the package, we will follow the convention that updating from v0.x.y to v0.x.(y+1) (for example v0.1.15 to v0.1.16) should not break your code, unless you are using internal/undocumented features of the code, while updating from `v0.x.y` to `v0.(x+1).y` might break your code, though we will try to add deprecation warnings when possible, such as for simple cases where the name of a function changes.

Note that as of Julia v1.5, in order to see deprecation warnings you will need to start Julia with `julia --depwarn=yes` (previously they were on by default). Please run your code like this before upgrading between minor versions of the code (for example from v0.1.41 to v0.2.0).

After we release v1 of the package, we will start following [semantic versioning](https://semver.org).

ITensors v0.3.22 Release Notes
==============================

Bugs:

Enhancements:

- `contract(::MPO, ::MPS)` rrule (#1025)

ITensors v0.3.21 Release Notes
==============================

Bugs:

Enhancements:

- Document the write_path option for dmrg (df2a665)
- Bump compats Functors 0.4, NDTensors 0.1.45 (#1031)
- Fix auto fermion system for non-QN tensors (#1027)
- MPOs constructed from sums of one-site operators now have bond dimension 2 (#1015)
- Update readwrite test (#1007)
- Generalize `inner(::MPO, ::MPS, ::MPO, ::MPS)` to other tag/prime conventions (#995)
- Fix variable declaration warnings (#994)
- Refactor (#980)
- Add Proj0, Proj1 aliases for ProjUp, ProjDn (3d84a5e)
- Define F operator for Qudit and Boson (#977)
- HDF5 Support for Diag Storage (#976)
- Allow user to set the path for disk (#975)

ITensors v0.3.20 Release Notes
==============================

Bugs:

- Fix bug contracting rectangular Diag with Dense (#970)

Enhancements:

- `site_range` keyword for `truncate!` to only truncate part of an MPS (#971)
- Fix issue with tolerance in `lognorm` when checking that the inner product is real (#973)

ITensors v0.3.19 Release Notes
==============================

Bugs:

- Fix bug in MPO(::OpSum, ...) when on-site operators have no blocks (#963)

Enhancements:

- Allow specifying non-uniform link dimensions in MPS constructors (#951)
- Add splitblocks keyword argument to `MPO(::OpSum, ...)` constructor, which defaults to `splitblocks=true` (#963)
- Allow specifying the element type of the output MPO with MPO(::Type, ::OpSum, ...), for example MPO(Float32, opsum, sites) to use single precision (#963)
- Improve overloading interface for "Qudit"/"Boson" types, overload `ITensors.op(::OpName"new_op", ::SiteType"Qudit", d::Int)` (#963)
- Fix typo in contraction sequence optimization code for some cases with empty indices (#965)
- Add keyword arguments support for `ITensors.state` (#964)
- Document and tweak the tolerance of the `nullspace` function (#960)
- Improve functionality for transferring data between CPU and GPU (#956)
- Fix `expect` and `correlation_matrix` for MPS on GPU (#956)
- Make sure QR and SVD preserve element type (i.e. single precision) in more cases (#956)
- Remove Sweeps object in examples and docs (#959)
- Pass kwargs through to truncate in Dense factorizations (#958)
- Optimize apply `rrule` for MPS/MPO by improving contraction sequence when contracting forward and reverse MPS/MPO (#955)
- Simplify the `rrule`s for priming and tagging MPS/MPO (#950)

ITensors v0.3.18 Release Notes
==============================

Bugs:

- Extend `apply(::MPO, ::MPO)` to `apply(::MPO, ::MPO, ::MPO...)` (#949)
- Fix AD for `apply(::MPO, ::MPO)` and `contract(::MPO, ::MPO)` (#949)
- Properly use element type in `randomMPS` in the 1-site case (b66d1b7)
- Fix bug in `tr(::MPO)` rrule where the derivative was being multiplied twice into the identity MPO (b66d1b7)
- Fix directsum when specifying a single `Index` (#930)
- Fix bug in loginner when inner is negative or complex (#945)
- Fix subtraction bug in `OpSum` (#945)

Enhancements:

- Define "I" for Qudit/Boson type (b66d1b7)
- Only warn in `inner` if the result is `Inf` or `NaN` (b66d1b7)
- Make sure `randomITensor(())` and `randomITensor(Float64, ())` returns a Dense storage type (b66d1b7)
- Define `isreal` and `iszero` for ITensors (b66d1b7)
- Project element type of ITensor in reverse pass of tensor-tensor or scalar-tensor contraction (b66d1b7)
- Define reverse rules for ITensor subtraction and negation (b66d1b7)
- Define `map` for ITensors (b66d1b7)
- Throw error when performing eigendecomposition of tensor with NaN or Inf elements (b66d1b7)
- Fix `rrule` for `MPO` constructor by generalizing the `rrule` for the `MPS` constructor (#946)
- Forward truncation arguments to more operations in `rrule` for `apply` (#945)
- Add rrules for addition and subtraction of MPOs (#935)

ITensors v0.3.17 Release Notes
==============================

Bugs:

Enhancements:

- Add Zp as alias for operator Z+, etc. (#942)
- Export diag (#942)

ITensors v0.3.16 Release Notes
==============================

Bugs:

Enhancements:

- Define `nullspace` for ITensors (#929)

ITensors v0.3.15 Release Notes
==============================

Bugs:

Enhancements:

- Fix `randomMPS` and `svd` for `Float32`/`ComplexF32` (#926)

ITensors v0.3.14 Release Notes
==============================

Bugs:

Enhancements:

- Add backend `alg="directsum"` for MPS/MPO addition (#925)
- Add `alg="naive"` for MPO contraction (#925)
- Add `svd`/`eigen` option `cutoff<0` or `cutoff=nothing`, indicating that no truncation should be performed based on a cutoff (previously you could only specify `cutoff=0.0` which still truncated eigenvalues of 0) (#925)
- Fixes an issue that `mindim` wasn't be used in `eigen` (#925)
- Remove `OpSum` in favor of `Ops.OpSum` (#920)

ITensors v0.3.13 Release Notes
==============================

Bugs:

Enhancements:

- Implement `min_blockdim` keyword for blocksparse SVD (#923)
- Add support for non-zero flux MPOs to OpSum (#918)

ITensors v0.3.12 Release Notes
==============================

Bugs:

Enhancements:

- Fix `svd` and `qr` for empty input left or right indices (#917)
- Add support for defining MPOs from operators represented as matrices (#904)

ITensors v0.3.11 Release Notes
==============================

Bugs:

Enhancements:

- Introduce `removeqn` function for removing a specified quantum number (#915)
- Non-Hermitian `dmrg` (#913)
- Clean up QN `svd` code in `ITensors` by handling QN blocks better in `NDTensors` (#906)

ITensors v0.3.10 Release Notes
==============================

Bugs:

Enhancements:

- Update installation instructions for Julia 1.7.

ITensors v0.3.9 Release Notes
=============================

Bugs:

Enhancements:

- Haar random unitary gate and generalize identity operator to arbitrary number of sites (#903).
- Improve error messages for op.
- Return the original MPS/MPO when normalizing a zero MPS/MPO (#901).
- Allow Matrix representations for operators in `expect` and `correlation_matrix` (#902).

ITensors v0.3.8 Release Notes
=============================

Bugs:

Enhancements:

- Increase maximum TagSet size to 16 characters (#882)

ITensors v0.3.7 Release Notes
=============================

Bugs:

- Fix for performance issue when applying gates that skip sites (#900).

Enhancements:

ITensors v0.3.6 Release Notes
=============================

Bugs:

- Fix bug in `op(opname, s::Vector{<:Index})` and `op(s::Vector{<:Index}, opname)`.

Enhancements:

ITensors v0.3.5 Release Notes
=============================

Bugs:

Enhancements:

- Generalize `op` to handle `Matrix`/`String` inputs more generically (#899)

ITensors v0.3.4 Release Notes
=============================

Bugs:

Enhancements:

- Simplify rrules for Index manipulation of ITensors (#888)
- Add some helper functions like converting element types of ITensors (#898)
  - `cu([A, B])` -> `[cu(A), cu(B)]` (same for `cpu`).
  - `cu([[A, B], [C]])` -> `[[cu(A), cu(B)], [cu(C)]]` (same for `cpu`).
  - `convert_eltype(T::Type, A::ITensor)` - convert the element type of an ITensor to `T`.
  - `convert_leaf_eltype(T, A::MPS)` - convert the element types of the ITensors of an MPS/MPO.
  - `convert_leaf_eltype(T, [[A, B], C])` - convert the element types of ITensors `A`, `B`, `C` in a nested data structure (useful for layered gate structures used in PastaQ).
  - `contract(A::MPS)` - contract the ITensors of an MPS/MPO into an ITensor (previously we used `prod` for that but I think using ` contract` is clearer).
  - `array(A::ITensor, i::Index, j::Index, ...)` - convert the ITensor to an Array, first permuting into the Index ordering `i, j, ...`. Previously I used `array(permute(A, i, j, ...))` for this but this is more convenient.
  - `A(x)` as a simpler syntax for `apply(A::ITensor, x::ITensor)`, treating `A` as an operator from unprimed to primed indices. I've already defined this syntax for `MPO` and `MPS` and I think it is pretty nice. I was holding off on doing this for a while to see if there might be a better meaning for `A(B)` but 
  - Define `complex`, `real`, `imag`, and `conj` for MPS/MPO by applying them to the ITensors of the MPS/MPO. Maybe there is a better meaning for these, as in the MPS that is the real part of the MPS defined as a state?

ITensors v0.3.3 Release Notes
=============================

Bugs:

Enhancements:

- Add `copy` for `AbstractProjMPO` (#895)

ITensors v0.3.2 Release Notes
=============================

Bugs:

Enhancements:

- Introduce `set_nsite!` generic `AbstractProjMPO` function (#894)
- Factorize out `contract(::ProjMPO, ::ITensor)` (#893)

ITensors v0.3.1 Release Notes
=============================

Bugs:

Enhancements:

- Introduce `Algorithm` type for selecting algorithm backends (#886)

ITensors v0.3.0 Release Notes
=============================

Bugs:

Enhancements:

- Introduce `apply(::MPO, ::MPO)` (#880)
- Make automatic differentiation work for `contract(::ITensor...)` (#878)
- Deprecate automatically making indices match in `inner` and `outer` (#877)
  - Add test for `apply(::MPO, ::MPS) = noprime(contract(::MPO, ::MPS))` and lazy version `Apply(::MPO, ::MPS)`.
  - Define `isapprox(::AbstractMPS, ::AbstractMPS)`.
- correlation_matrix sites keyword (#868)
  - Implement non-contiguous sites for correlation_matrix
- rrule for MPS(Vector{::ITensor}) (#865)
  - `rrule` for constructing an `MPS` from a `Vector{ITensor}`.
  - Improve `op(::OpName ,::SiteType"Qudit")` for handling two-body ops.
  - Add support for storing a `Function` in an `op` in the format `(f, opame, support, (params...))`.
- Fix expect for complex MPS (#867)
- Get some AD working for LazyApply and Ops (#859)
- + and - in the op system (#857)
- Rename expect site_range keyword to sites (#858)
  - Allow more general sites collections to be passed including single site number that maps to scalar outputs.
  - Add ishermitian for ITensors
  - Improve handling of types and non-Hermitian operators in expect
  - Define ITensor transpose
- Improve Sweeps constructors with keyword arguments and default init (#856)
- rrules for apply(U, ::MPO), `(::MPO * ::MPO)`, `tr(::MPO)` (#852)
- Unification of PastaQ.gate and `ITensors.op`, new `OpSum` algebra functions (#843)
- Change minimal required Julia version from 1.3 to 1.6 (#849)
  - Add default `maxdim=typemax(Int)` in `dmrg`.

ITensors v0.2.16 Release Notes
==============================

Bugs:

- Fix `inner` `MPS` `rrule` for complex and QNs, add tests (#836)
- Fix differentation of apply(::ITensor, ::ITensor) (#831)
- Fix string indexing when setting elements (#826)
    - Fix string indexing when setting elements, such as `T[i => "Up"] = 1.0`.
    - Change `ITensor` `rrule` constructor signature from `typeof(ITensor)` to `Type{ITensor}`.

Enhancements:

- Define `diag(::Tensor)`, `diag(::ITensor)` (#837)
- Support indexing notation A[Up] (#839)
- Allow dividing by scalar ITensor (#838)
- Add `normalize[!](::MPS/MPO)` (#820)
- More flexible `state` syntax (overloading and calling) (#833)
    - `state` defined with Index now should return an `ITensor`, for consistency with how `op` definitions work (this PR supports backwards compatibility for the now-deprecated syntax which returns a Vector that gets automatically converted to an ITensor
).
    - Allow syntax like `state("Up", Index(2, "S=1/2"))` (previously only `state(Index(2, "S=1/2"), "Up")` worked).
- Define inner(::ITensor, ::ITensor) (#835)
- Add more compact kwarg sweeps syntax for DMRG (#834)
- Fix prime/tag rrule for MPS/MPO (#830)
- Simplify Dense AutoMPO Backend, output lower triangular MPO in dense case (#828)
- Fix `ITensor` `rrule` with `Array` reshaping (#824)
- Add Riemannian optimization example

ITensors v0.2.15 Release Notes
==============================

- Handle non-Hermitian `correlation_matrix` properly (#817)
- Fix ElecK state definitions in example.

ITensors v0.2.14 Release Notes
==============================

- Fix ITensor rrule and generalize ITensor to Array conversion (#818)
- Fix apply and inner rrules for complex and QN conserving MPS (#816)
- Change ordering that `op` definitions get called (#816)

ITensors v0.2.13 Release Notes
==============================

- Fix eltype promotion dividing ITensor by scalar (#813)
- Make getindex on EmptyStorage return EmptyNumber (#812)
- Add variational circuit optimization (#811)
- Expand ITensor development guide (#809)
- Add issue templates for all subdir packages (#808)
- Change randomMPS bond dim error to warning (#806)
- Add docs for enabling debug checks (#801)
- Remove ignore comments for JuliaFormatter (#799)
- Improve docstrings for apply and ITensor constructors (#797)
- FAQ on the relationship of ITensor to other tensor libraries (#795)
- Move ITensorVisualizationCore into ITensors module (#787)

ITensors v0.2.12 Release Notes
==============================

- Use registered subdir version of NDTensors (#780)

ITensors v0.2.11 Release Notes
==============================

- Fix bug when slicing a Dense Tensor with mixed ranges and integers (#775)
- Fix rrule for Array to ITensor constructor when forward and reverse indices are in different orders (#773)
- Fix MPO OpSum for operator names with unicode (#767)
- Lazy Op system (#769)
- Add ITensorNetworkMap (#764)
- Fix conj and scalar multiplication of EmptyStorage (#756)
- Fix the link for the qudit sitetype (#762)

ITensors v0.2.10 Release Notes
==============================

- Add ChainRules rules for basic reverse mode AD operations with ITensors (#761)

ITensors v0.2.9 Release Notes
==============================

- Fix test truncation test for Julia 1.7 (#755)
- Fix contraction ordering in correlation matrix (#754)

ITensors v0.2.8 Release Notes
==============================

- Fix bug in `permute` (and therefore `indpairs` and `tr(::ITensor)`) (#750)
- Add support for multisite operators and passing parameters into operators in `OpSum` (no support in `MPO` construction yet) (#749)
- Fix subtraction of term from `OpSum`. Add more unicode operator name aliases (#748)

ITensors v0.2.7 Release Notes
==============================

- Fix bug in threaded block sparse contraction, add CI tests for threading (#746)
- Limit `maxdim` in default ("densitymatrix") contract `MPO * MPS` code (#744)
- Add highlighting to ITensor paper bibtex in README (#738) 
- Generalize `Array` -> `ITensor` constructor to allow `AbstractArray` (#737)
- Fix dispatch issue when indices input into ITensor constructor had abstract element types like `Vector{Index}` (#737)
- Pass kwargs in MPO contract (#733)
- Use Compat to make blas_get_num_threads simpler (#731)
- Generalize ITensor constructors to allow mixtures of collections of indices (`Tuple`s and `Vector`s) (#728)
- Add some more ITensor constructors like `ITensor([2.3])` and `ITensor(2.3, QN(), i', dag(i))` (#728)

ITensors v0.2.6 Release Notes
==============================

- Add Qudit site type with QNs as well as Boson alias (#727) 
- Tighten accuracy cutoff for OpSum/AutoMPO (#726)
- Add support for complex data written by C++ ITensor for block sparse tensors (#724)

ITensors v0.2.5 Release Notes
==============================

- Fixed bug involving missing default case for state function (#719)
- Add support for reading complex ITensors written from C++ (#720)
- Fix HDF5 read compatilibity between ITensors v0.1 and v0.2 (#715) (@tschneider) 
- Improve inference in NDTensors contraction and start writing a new precompile file (off by default) (#655)

ITensors v0.2.4 Release Notes
==============================

- Fix state function when overloading version that accepts an Index (#711)
- Started work on FAQs (#709)
- Add denseblocks definition for Diag storage (#710)
- Code examples about making an array from an ITensor, QR with postive R, and sampling an MPS (#705)
- Prepend site type functions with ITensors to make it clearer to users how they should write their own overloads (#704)
- Fix issue with Index arrow for QN case of `correlation_matrix` (#702)
- Add `sim(::Pair{<:Index})` (#701)
- Add `norm(::EmptyStorage)` (#699) 

ITensors v0.2.3 Release Notes
==============================

- Add `denseblocks` to convert from `DiagBlockSparse` to `BlockSparse` storage (PR #693).
- Make A == B return false if ITensors A and B have different indices (PR #690).

ITensors v0.2.2 Release Notes
==============================

- Make Index non-broadcastable so you can do: `i = Index(2); i .^ (0, 1, 2)` (PR #689).
- Add interface `contract([A, [B, C]])` for recursively contracting a tensor network tree, equivalent to `contract([A, B, C]; sequence=[1, [2, 3]])` (PR #686).
- Allow plain integer tags, such as `Index(2, "1")` (PR #686).

ITensors v0.2.1 Release Notes
==============================

- Fix MPS product state constructor to use new state function system, and fix some incorrect site type overloads (for Fermion and S=1/2) (PR #685)
- Improve and update documentation in various places

ITensors v0.2.0 Release Notes
==============================

Breaking changes:
-----------------

- Change QN convention of the Qubit site type to track the total number of 1 bits instead of the net number of 1 bits vs 0 bits (i.e. change the QN from +1/-1 to 0/1) (PR #676).
- Remove `IndexVal` type in favor of `Pair{<:Index}`. `IndexVal{IndexT}` is now an alias for `Pair{IndexT,Int}`, so code using `IndexVal` such as `IndexVal(i, 2)` should generally still work. However, users should change from `IndexVal(i, 2)` to `i => 2` (PR #665).
- Rename the `state` functions currently defined for various site types to `val` for mapping a string name for an index to an index value (used in ITensor indexing and MPS construction). `state` functions now return single-index ITensors representing various single-site states (PR #664).
- `maxlinkdim(::MPO/MPS)` returns a minimum of `1` (previously it returned 0 for MPS/MPO without and link indices) (PR #663).
- The `NDTensors` module has been moved into the `ITensors` package, so `ITensors` no longer depends on the standalone `NDTensors` package. This should only effect users who were using both `NDTensors` and `ITensors` seperately. If you want to use the latest `NDTensors` library, you should do `using ITensors.NDTensors` instead of `using NDTensors`, and will need to install `ITensors` with `using Pkg; Pkg.add("ITensors")` in order to use the latest versions of `NDTensors`. Note the current `NDTensors.jl` package will still exist, but for now developmentof `NDTensors` will occur within `ITensors.jl` (PR # 650).
- `ITensor` constructors from collections of `Index`, such as `ITensor(i, j, k)`, now return an `ITensor` with `EmptyStorage` (previously called `Empty`) storage instead of `Dense` or `BlockSparse` storage filled with 0 values. Most operations should still work that worked previously, but please contact us if there are issues (PR #641).
- ITensors now store a `Tuple` of `Index` instead of an `IndexSet` (PR #626).
- The ITensor type no longer has separate field `inds` and `store`, just a single field `tensor` (PR #626).
- The `IndexSet{T}` type has been redefined as a type alias for `Vector{T<:Index}` (which is subject to change to some other collection of indices, and likely will be removed in ITensors v0.3). Therefore it no longer has a type parameter for the number of indices, similar to the change to the `ITensor` type. If you were using the plain `IndexSet` type, code should generally still work properly. In general you should not have to use `IndexSet`, and can just use `Tuple` or `Vector` of `Index` instead, such as `is = (i, j, k)` or `is = [i, j, k]`. Priming, tagging, and set operations now work generically on those types (PR #626).
- ITensor constructors from Array now only convert to floating point for `Array{Int}` and `Array{Complex{Int}}`. That same conversion is added for QN ITensor constructors to be consistent with non-QN versions (PR #620) (@mtfishman).
- The tensor order type paramater has been removed from the `ITensor` type, so you can no longer write `ITensor{3}` to specify an order 3 ITensor (PR #591) (@kshyatt).

Deprecations:
-----------------

- `Index(::Int)` and `getindex(i::Index, n::Int)` are deprecated in favor of `Pair` syntax (using `i => 3` instead of `i(3)` or `i[3]`) (PR #665).
- `Base.iterate(i::Index, state = 1)` is deprecated in favor of `eachindval` (PR #665).
- Deprecate `MPO(::MPS)` in favor of `outer(::MPS, ::MPS)` (PR #663).
- Add `OpSum` as alternative (preferred) name to `AutoMPO` (PR #663).
- `noise!(::Sweeps, ...)` and `cutoff!(::Sweeps, ...)` are deprecated in favor of `setnoise!` and `setsweeps!` (PR #624).
- `emptyITensor(Any)` is deprecated in favor of `emptyITensor()` (PR #620).
- `store` is deprecated in favor of `storage` for getting the storage of an ITensor. Similarly `ITensors.setstore[!]` -> `ITensors.setstorage[!]` (PR #620).

Bug fixes and new features:
-----------------

- Fix bug when taking the exponential of a QN ITensor with missing diagonal blocks (PR #682).
- Generalize indexing to with `end` to allow arithmetic such as `A[i => end - 1, j => 2]` (PR #679).
- Allow Pair inputs in `swaptags` and `swapinds`, i.e. `swaptags(A, "i" => "j")` and `swapinds(A, i => j)` (PR #676).
- Fix negating of QN ITensor (PR #672) (@emstoudenmire).
- Add support for indexing into ITensors with strings, such as `s = siteind("S=1/2"); T = randomITensor(s); T[s => "Up"]` (indices must have tags that have `val` overloads) (PR #665).
- Add `eachindval(::Index)` and `eachval(::Index)` for iterating through the values of an Index (PR #665).
- Rename the `state` functions currently defined for various site types to `val` for mapping a string name for an index to an index value (used in ITensor indexing and MPS construction). `state` functions now return single-index ITensors representing various single-site states (PR #664).
- Out-of-place broadcasting on MPS that maps ITensors to ITensors like `2 .* psi` will return an MPS (previously it returned a `Vector{ITensor}`) (PR #663).
- Add `outer(psi::MPS, phi::MPS)::MPO -> |psi><phi|` to do an outer product of two `MPS` and `projector(psi::MPS)::MPO -> |psi><psi|`, deprecate `MPO(::MPS)` in favor of these (PR #663).
- Redefine `eltype(::MPS/MPO)` as ITensor, i.e. the actual element type of the storage data, thinking about the MPS as a `Vector{ITensor}`. This matches better with how `eltype` is used in general in Julia, and therefore helps MPS by more compatible with what generic Julia code expects (like broadcasting, where an issue related to this came up). A new function `promote_itensor_eltype(::MPS/MPO)` returns the promoted element types of the ITensors of the MPS to replace the old functionality (PR #663).
- Fix a bug in broadcasting an MPS/MPO such as `psi .*= 2` where previously it wasn't expanding the orthogonality limits to the edge of the system. Also now out-of-place broadcasting like `2 .* psi` will return an MPS (previously it returned a Vector{ITensor}) (PR #663).
- Fix bug in `MPO(::AutoMPO, ::Vector{<:Index})` where for fermionic models the AutoMPO was being modified in-place (PR #659) (@mtfishman).
- Add `Array` to `MPS` constructor (PR #649).
- Faster noise term in DMRG (PR #623)
- Add `expect` and `correlation_matrix` to help measure expectation values and correlation matrices of MPS (PR #622).
- Add `real`, `imag`, `conj` for ITensors (PR #621).
- ITensor constructors from Array now only convert to floating point for `Array{Int}` and `Array{Complex{Int}}`. That same conversion is added for QN ITensor constructors to be consistent with non-QN versions (PR #620) (@mtfishman).
- New ITensor constructors like `itensor(Int, [0 1; 1 0], i, j)` to specify the exact element type desired (PR #620) (@mtfishman).
- Make QN ITensor constructor error in case of no blocks (PR #617).
- Speed up `randomITensor` with `undef` constructor (PR #616) (@emstoudenmire).
- Fix definition of `Adagdn` for Electron (PR #615) (@emstoudenmire).
- Add `onehot` as new name for `setelt` (PR #615) (@emstoudenmire).
- Support `randomMPS(ComplexF64,s,chi)` (PR #615) (@emstoudenmire).
- Make TagSet code cleaner and more generic, and improve constructor from String performance (PR #610) (@saolof).
- Define some missing methods for AbstractMPS broadcasting (#609) (@kshyatt).
- Make ops less strict about input (PR #602) (@emstoudenmire).
- Add support for using `end` in `setindex!` for ITensors (PR #596) (@mtfishman).
- Contraction sequence optimization (PR #589) (@mtfishman).

ITensors v0.1.41 Release Notes
==============================
- Add "Qubit" site type (alias for "S=1/2"), along with many quantum gate definitions (PR #592) (@emstoudenmire).

ITensors v0.1.40 Release Notes
==============================
- Remove eigen QN fix code (simplify ITensors eigen code by handling QNs better in NDTensors eigen) (PR #587) (@emstoudenmire).
- More general polar decomposition that works with QNs (PR #588) (@mtfishman).
- Bump to v0.1.28 of NDTensors, which includes some bug fixes for BlockDiag storage and makes ArrayInterface setindex compatiblity more general (NDTensors PR #68) (@mtfishman).

ITensors v0.1.39 Release Notes
==============================
- Add Pauli X,Y,Z to S=1/2 site type (PR #576) (@emstoudenmire).
- Add truncation error output to DMRG (PR #577) (@emstoudenmire).
- Bump StaticArrays version to v1.0 (PR #578) (@mtfishman).
- Fix orthogonalize when there are missing MPS link indices (PR #579) (@mtfishman). 
- Simplify MPO * MPO contraction and make more robust for MPOs with multiple site indices per tensor (PR #585) (@mtfishman).

ITensors v0.1.38 Release Notes
==============================
- New MPS/MPO index manipulation interface (PR #575) (@mtfishman).
- Add support for `inner(::MPS, ::MPO, ::MPS)` with multiple siteinds per tensor (PR #573) (@mtfishman).
- Fix `MPO*MPS`, `MPO*MPO` for system sizes 1 and 2 (PR #572) (@mtfishman).
- Add generic support for scalar ITensor contraction (PR #569) (@mtfishman).
- Fix and add tests for printing QN diag ITensors (PR #568) (@mtfishman).

ITensors v0.1.37 Release Notes
==============================
- Bump to NDTensors v0.1.23 which fixes a bug in block sparse multithreading when a block sparse tensor contraction results in a tensor with no blocks (PR #565) (@mtfishman).

ITensors v0.1.36 Release Notes
==============================
- Bump to v0.1.22 of NDTensors which introduces block sparse multithreading. Add documentation and examples for using block sparse multithreading (PR #561) (@mtfishman).
- Make dmrg set ortho center to 1 before starting (PR #562) (@emstoudenmire).

ITensors v0.1.35 Release Notes
==============================
Closed issues:

- Should we define iterate for TagSet? (#542)
- AutoMPO slower than expected (#555)

Merged pull requests:

- Implement iterate for TagSet (#553) (@tomohiro-soejima)
- Add check for Index arrows for map! (includes sum and difference etc) (#554) (@emstoudenmire)
- Optimize AutoMPO (#556) (@mtfishman)
- Add checks for common site indices in DMRG, dot, and inner (#557) (@mtfishman)
- Fix and Improve DMRGObserver Constructor (#558) (@emstoudenmire)
- Update HDF5 to versions 0.14, 0.15 (#559) (@emstoudenmire)

ITensors v0.1.34 Release Notes
==============================
* Allow operator names in the `op` system that are longer than 8 characters (PR #551).

ITensors v0.1.33 Release Notes
==============================
* Fix bug introduced in v0.1.32 involving inner(::MPS, ::MPS) if the MPS have more than one site Index per tensor (PR #549).

ITensors v0.1.32 Release Notes
==============================
* Update to NDTensors v0.1.21, which includes a bug fix for scalar-like tensor contractions involving mixed element types (NDTensors PR #58).
* Docs for observer system and DMRGObserver (PR #546).
* Add `ITensors.@debug_check`, `ITensors.enable_debug_checks()`, and `ITensors.disable_debug_checks()` for denoting that a block of code is a debug check, and turning on and off debug checks (off by default). Use to check for repeated indices in IndexSet construction and other checks (PR #547).
* Add `index_id_rng()`, an RNG specifically for generating Index IDs. Set the seed with `Random.seed!(index_id_rng(), 1234)`. This makes the random stream of number seperate for Index IDs and random elements, and helps avoid Index ID clashes with reading and writing (PR #547).
* Add back checking for proper QN Index directions in contraction (PR #547).

ITensors v0.1.31 Release Notes
==============================
* Update to NDTensors v0.1.20, which includes some more general block sparse slicing operations as well as optimizations for contracting scalar-like (length 1) tensors (NDTensors PR #57).
* Add flux of IndexVal functionality which returns the QN multiplied by the direction of the Index. Make `qn` consistently return the bare QN. Might be breaking for people who were calling `qn(::IndexVal)` and related functions, since now it consistently returns the QN not modified by the Index direction (PR #543).
* Introduce `splitblocks` function for Index, ITensor and MPS/MPO. This splits the QNs of the specified indices into blocks of size 1 and drops nonzero blocks, which can make certain tensors more sparse and improve efficiency. This is particularly useful for Hamiltonian MPOs. Thanks to Johannes Hauschild for pointing out this strategy (PR #540).
* Add Ising YY and ZZ gates to qubits examples (PR #539).

ITensors v0.1.30 Release Notes
==============================
* Update to NDTensors v0.1.19, which includes various block sparse optimizations. The primary change is switching the block-offset storage from a sorted vector to a dictionary for O(1) lookup of the offsets. Note this may be a slightly breaking change for users that were doing block operations of block sparse tensors since now blocks have a special type Block that stores a tuple of the block location and the hash (NDTensors PR #54 and ITensors PR #538).

ITensors v0.1.29 Release Notes
==============================
* Add global flag for combining before contracting QN ITensors, control with enable/disable_combine_contract!(). This can speed up the contractions of high order QN ITensors (PR #536).
* Fix bug when using "end" syntax when indexing ITensors where the Index ordering doesn't match the internal ITensor Index order (PR #537).
* Increase NDTensors to v0.1.18, which includes a variety of dense and sparse contraction optimizations.

ITensors v0.1.28 Release Notes
==============================
* Add support for setting slices of an ITensor (PR #535).
* Add bond dimension maximum in addition of MPS/MPO based on sums of bond dimensions of original MPS/MPO (PR # 535).
* Add TBLIS contraction support. When TBLIS.jl is installed, the command "using TBLIS" turns on TBLIS support. enable_tblis!() and disable_tblis!() also turn TBLIS backend on and off (PR #533).
* Add DMRG and contraction examples of using TBLIS contraction backend (PR #533).

ITensors v0.1.27 Release Notes
==============================
* Use LAPACK's gesdd by default in SVD (PR #531).

ITensors v0.1.26 Release Notes
==============================
* Introduce a density matrix algorithm for summing arbitrary numbers of MPS/MPO (non-QN and QN) (PR #528).
* Introduce @preserve_ortho macro, which indicates that a block of code preserves the orthogonality limits of a specified MPS/MPO or set of MPS/MPO (PR #528).
* Introduce the ortho_lims(::MPS/MPO) function, that returns the orthogonality limits as a range (PR #528).
* Improves the (::Number * ::MPS/MPO) function by ensuring the number scales an MPS/MPO tensor within the orthogonality limits (PR #528).
* Improve functionality for making an MPO that is a product of operators. In particular, MPO(s, "Id") now works for QN sites, and it adds notation like: MPO(s, n -> isodd(n) ? "S+" : "S-") (PR #528).
* Add SiteType and op documentation.
* Add unexported function ITensors.examples_dir to get examples directory.

ITensors v0.1.25 Release Notes
==============================
* Introduce imports.jl to organize import statements (PR #511).
* Add TRG and isotropic CTMRG examples (PR #511).
* Add example for 2D Hubbard model with momentum conservation around the cylinder (PR #511).
* Fix fermion string issue (PR #519)

ITensors v0.1.24 Release Notes
==============================
* Generalize `tr(::MPO)` for MPOs with more an one pair of site indices per site (PR #509)
* Add `tr(::ITensor)` to trace pairs of indices of an ITensor (PR #509)
* Add stacktrace to warn tensor order (PR #498)

ITensors v0.1.23 Release Notes
==============================
* Add lastindex(A::ITensor, n::Int) to define A[end, end]. (PR #495)
* Define hastags(A::ITensor, ts) and related functions. (PR #495)
* Fix some broadcasting. Add Hadamard product and division. (PR #495)
* Add tr(::MPO) (PR #492)
* Add docstrings and docs for apply(::Vector{ITensor}, ::MPS) (PR #492)
* Add docstrings for IndexSet set functions like commoninds, uniqueinds, etc. (PR #492)

ITensors v0.1.22 Release Notes
==============================
* Add MPS/MPO circuit evolution with the apply function (PR #480)
* Improve MPS docs (PR #488)
* dense function for MPS/MPO (PR #483)
* MPO sampling (PR #486)
* Allow conserving Sz up or down in Fermion type (PR #482)
* Docstrings for siteinds method (PR #481)
* movesites function for MPS/MPO for permuting sites (PR #477)
* New Order value type for representing the order of a tensor at compile time (PR #475)
* Add generic "F" operator for non-fermion site types (PR #469)

ITensors v0.1.21 Release Notes
==============================
* Add parity conservation to S=1/2 sitetype (PR #467)
  * Add "ProjUp" and "ProjDn" operator definitions to S=1/2 site type.
  * Change QN name "Pf" to "NfParity"
  * Add keyword arguments to choose the QN names when making siteinds.
* Add ! as not syntax (PR #471)
  * Add @ts_str macro for TagSet construction
* Add Sweeps constructor from matrix of parameters (PR #472)
* Add examples of input files (PR #473)
  * Add native ITensors argument parsing with the argsdict() function.
  * Add examples of using ITensors with input files and ArgParse.jl and argsdict().

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

