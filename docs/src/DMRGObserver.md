# DMRGObserver

A DMRGObserver is a type of [observer](@ref observer) which
offers certain useful, general purpose capabilities 
for DMRG calculations such as measuring custom 
local observables at each step and stopping DMRG
early if certain energy convergence conditions are met.

In addition to the example code below, more detailed 
example code showing sample usage of DMRGObserver is included
in the ITensor source, in the file `1d_ising_with_observer.jl`
under the folder `examples/dmrg`.

## Sample Usage

In the following example, we have already made a Hamiltonian MPO `H`
and initial MPS `psi0` for a system of spins whose sites
have an associated "Sz" operator defined. We construct a 
`DMRGObserver` which measures "Sz" on each site at each
step of DMRG, and also stops the calculation early if
the energy no longer changes to a relative precision of 1E-7.

```
Sz_observer = DMRGObserver(["Sz"],sites,energy_tol=1E-7)

energy, psi = dmrg(H,psi0,sweeps,observer=Sz_observer)

for (sw,Szs) in enumerate(measurements(Sz_observer)["Sz"])
  println("Total Sz after sweep $sw = ", sum(Szs)/N)
end
```


## Constructors

```@docs
DMRGObserver(;energy_tol::Float64,minsweeps::Int)
DMRGObserver(ops::Vector{String},sites::Vector{<:Index};energy_tol::Float64,minsweeps::Int)
```

## Methods

```@docs
measurements(::DMRGObserver)
DMRGMeasurement
energies(::DMRGObserver)
```

