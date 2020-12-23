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

