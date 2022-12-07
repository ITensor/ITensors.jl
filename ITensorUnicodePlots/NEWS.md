This file is a (mostly) comprehensive list of changes made in each release of ITensorUnicodePlots.jl. For a completely comprehensive but more verbose list, see the [commit history on Github](https://github.com/ITensor/ITensors.jl/commits/main/ITensorUnicodePlots).

While we are in v0.x of the package, we will follow the convention that updating from v0.x.y to v0.x.(y+1) (for example v0.1.15 to v0.1.16) should not break your code, unless you are using internal/undocumented features of the code, while updating from `v0.x.y` to `v0.(x+1).y` might break your code, though we will try to add deprecation warnings when possible, such as for simple cases where the name of a function changes.

Note that as of Julia v1.5, in order to see deprecation warnings you will need to start Julia with `julia --depwarn=yes` (previously they were on by default). Please run your code like this before upgrading between minor versions of the code (for example from v0.1.41 to v0.2.0).

After we release v1 of the package, we will start following [semantic versioning](https://semver.org).

ITensorUnicodePlots v0.1.3 Release Notes
========================================

Bugs

Enhancements

- Update compats (#1031)

ITensorUnicodePlots v0.1.2 Release Notes
========================================

Bugs

Enhancements

- Drop explicit dependency on ITensors

ITensorUnicodePlots v0.1.1 Release Notes
========================================

- Remove newlines from unicode visualization (#819)

ITensorUnicodePlots v0.1.0 Release Notes
========================================

- Register ITensorUnicodePlots package, code in ITensors.jl repository
