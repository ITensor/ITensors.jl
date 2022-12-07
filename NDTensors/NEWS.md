This file is a (mostly) comprehensive list of changes made in each release of NDTensors.jl. For a completely comprehensive but more verbose list, see the [commit history on Github](https://github.com/ITensor/ITensors.jl/commits/main/NDTensors).

While we are in v0.x of the package, we will follow the convention that updating from v0.x.y to v0.x.(y+1) (for example v0.1.15 to v0.1.16) should not break your code, unless you are using internal/undocumented features of the code, while updating from `v0.x.y` to `v0.(x+1).y` might break your code, though we will try to add deprecation warnings when possible, such as for simple cases where the name of a function changes.

Note that as of Julia v1.5, in order to see deprecation warnings you will need to start Julia with `julia --depwarn=yes` (previously they were on by default). Please run your code like this before upgrading between minor versions of the code (for example from v0.1.41 to v0.2.0).

After we release v1 of the package, we will start following [semantic versioning](https://semver.org).

NDTensors v0.1.45 Release Notes
===============================

Bugs:

-  HDF5 Support for Diag Storage (#976) 

Enhancements:

- Fix variable declaration warnings (#994)
- Bump compat to Functors 0.4 (#1031)
- Bump compat to Compat 4 (4facffe)
- Refactor and format (#980)

NDTensors v0.1.44 Release Notes
===============================

Bugs:

- Fix bug contracting rectangular Diag with Dense (#970)

Enhancements:

NDTensors v0.1.43 Release Notes
===============================

Bugs:

Enhancements:

- Improve functionality for transferring data between CPU and GPU by adding Adapt.jl compatibility (#956)
- Pass kwargs through to truncate in Dense factorizations (#958)

NDTensors v0.1.42 Release Notes
===============================

Bugs:

Enhancements:

- Define `map` for Tensor and TensorStorage (b66d1b7)
- Define `real` and `imag` for Tensor (b66d1b7)
- Throw error when trying to do an eigendecomposition of Tensor with Infs or NaNs (b66d1b7)

NDTensors v0.1.41 Release Notes
===============================

Bugs:

Enhancements:

- Fix `truncate!` for `Float32`/`ComplexF32` (#926)

NDTensors v0.1.40 Release Notes
===============================

Bugs:

Enhancements:

- Add support for `cutoff < 0` and `cutoff = nothing` for disabling truncating according to `cutoff` (#925)
- Define contraction of Diag with Combiner (#920)

NDTensors v0.1.39 Release Notes
===============================

Bugs:

Enhancements:

- Fix `svd` and `qr` for empty input left or right indices (#917)

NDTensors v0.1.38 Release Notes
===============================

Bugs:

Enhancements:

- Clean up QN `svd` code in `ITensors` by handling QN blocks better in `NDTensors` (#906)
- Clean up `outer` and add GEMM routing for CUDA (#887)

NDTensors v0.1.37 Release Notes
===============================

Bugs:

Enhancements:

- Add fallbacks for when LAPACK SVD fails (#885)

NDTensors v0.1.36 Release Notes
===============================

Bugs:

Enhancements:

- Change minimal required Julia version from 1.3 to 1.6 (#849)

NDTensors v0.1.35 Release Notes
===============================

Bugs:

Enhancements:

- Allow general AbstractArray as data of `Dense` storage `Tensor`/`ITensor` (#848)

NDTensors v0.1.34 Release Notes
===============================

Bugs:

Enhancements:

- Define `diag(::Tensor)`, `diag(::ITensor)` (#837) 

NDTensors v0.1.34 Release Notes
===============================

Bugs:

Enhancements:

- Fix eltype promotion when dividing Tensor by scalar (#813)

NDTensors v0.1.33 Release Notes
===============================

Bugs:

Enhancements:

- Use registered subdir version of NDTensors (#780)
