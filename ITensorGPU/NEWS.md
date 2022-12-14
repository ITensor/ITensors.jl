This file is a (mostly) comprehensive list of changes made in each release of ITensorGPU.jl. For a completely comprehensive but more verbose list, see the [commit history on Github](https://github.com/ITensor/ITensors.jl/commits/main/ITensorGPU).

While we are in v0.x of the package, we will follow the convention that updating from v0.x.y to v0.x.(y+1) (for example v0.1.15 to v0.1.16) should not break your code, unless you are using internal/undocumented features of the code, while updating from `v0.x.y` to `v0.(x+1).y` might break your code, though we will try to add deprecation warnings when possible, such as for simple cases where the name of a function changes.

Note that as of Julia v1.5, in order to see deprecation warnings you will need to start Julia with `julia --depwarn=yes` (previously they were on by default). Please run your code like this before upgrading between minor versions of the code (for example from v0.1.41 to v0.2.0).

After we release v1 of the package, we will start following [semantic versioning](https://semver.org).

ITensorGPU v0.0.7 Release Notes
===============================

Bugs:

Enhancements:

- Bump version compat for dependencies.

ITensorGPU v0.0.6 Release Notes
===============================

Bugs:

Enhancements:

ITensorGPU v0.0.5 Release Notes
===============================

Bugs:

Enhancements:

- Clean up `outer` and add GEMM routing for CUDA (#887)

ITensorGPU v0.0.4 Release Notes
===============================

Bugs:

Enhancements:

- `cu([[A, B], [C]])` -> `[[cu(A), cu(B)], [cu(C)]]` and same for cpu (#898).
- Allow cutruncate to work for Float32s (#897).

ITensorGPU v0.0.3 Release Notes
===============================

Bugs:

- Fix bugs in complex SVD on GPU (with and without truncations) (#871)

Enhancements:

- Remove some unnecessary contract code (#860)

ITensorGPU v0.0.2 Release Notes
===============================

Bugs:

- Remove unnecessary `CuDense` type equality definition (#823)

Enhancements:

ITensorGPU v0.0.1 Release Notes
===============================

Bugs:

Enhancements:

- Register ITensorGPU package, code in ITensors.jl repository
