# ITensors.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensors.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensors.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensors.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensors.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensors.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Installation instructions

The package can be added as usual through the package manager:

```julia
julia> Pkg.add("ITensors")
```

## Examples

````julia
using ITensors: ITensors, ITensor, Index
using LinearAlgebra: qr
using NamedDimsArrays: aligndims, unname
using Test: @test
i = Index(2)
j = Index(2)
k = Index(2)
a = randn(i, j)
@test a[j[2], i[1]] == a[1, 2]
@test a[j => 2, i => 1] == a[1, 2]
a′ = randn(j, i)
b = randn(j, k)
c = a * b
@test unname(c, (i, k)) ≈ unname(a, (i, j)) * unname(b, (j, k))
d = a + a′
@test unname(d, (i, j)) ≈ unname(a, (i, j)) + unname(a′, (i, j))
@test a ≈ aligndims(a, (j, i))
q, r = qr(a, (i,))
@test q * r ≈ a
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

