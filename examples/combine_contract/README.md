For QN ITensors (ITensors with block sparse structure), ITensors.jl
provides two "modes" for contracting them. The default mode is treating
them as block sparse multi-dimensional arrays and contracting them
"as is".

In the second mode, the ITensors are reshaped into block
sparse matrices before contracting. Making use of the QN information
in the tensors, the second mode can lead to fewer blocks being contracted,
which can lead to a speedup depending on the number of blocks, the sizes
of the blocks, the order of the ITensors being contracted, etc.

Run the example using parity symmetry as follows:
```julia
julia> include("main.jl")

julia> main(Nmax = 8)
#################################################
# order = 2
#################################################

Contract:
  2.410 μs (76 allocations: 5.70 KiB)

Combine then contract:
  48.111 μs (899 allocations: 92.98 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 4
#################################################

Contract:
  11.329 μs (311 allocations: 24.73 KiB)

Combine then contract:
  111.227 μs (1628 allocations: 182.69 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 6
#################################################

Contract:
  119.332 μs (2125 allocations: 200.11 KiB)

Combine then contract:
  284.701 μs (4019 allocations: 470.45 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 8
#################################################

Contract:
  1.291 ms (16988 allocations: 1.75 MiB)

Combine then contract:
  1.250 ms (18801 allocations: 1.75 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 10
#################################################

Contract:
  13.555 ms (138908 allocations: 16.22 MiB)

Combine then contract:
  6.233 ms (103211 allocations: 8.28 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 12
#################################################

Contract:
  165.222 ms (1261274 allocations: 141.49 MiB)

Combine then contract:
  37.353 ms (492436 allocations: 41.43 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 14
#################################################

Contract:
  2.256 s (10387274 allocations: 1.24 GiB)

Combine then contract:
  187.437 ms (2158837 allocations: 187.26 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 16
#################################################

Contract:
  28.637 s (104661017 allocations: 14.86 GiB)

Combine then contract:
  1.416 s (17813407 allocations: 1.32 GiB)

C_contract ≈ C_combine_contract = true

```

