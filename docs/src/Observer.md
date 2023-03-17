# [Observer System for DMRG](@id observer)

An observer is an object which can be passed to the ITensor DMRG
algorithm, to allow measurements to be performed throughout
the DMRG calculation and to set conditions for early stopping
of DMRG.

The only requirement of an observer is that it is a subtype 
of `AbstractObserver`. But to do something interesting, it
should also overload at least one the methods `measure!`
or `checkdone!`.

A general purpose observer type called [`DMRGObserver`](@ref) is
included with ITensors which already provides some
quite useful features. It accepts a list of strings naming
local operators to be measured at each step of DMRG, with
the results saved for later analysis. It also accepts an
optional energy precision, and stops a DMRG calculation early
if the energy no longer changes to this precision. For more
details about the [`DMRGObserver`](@ref) type, see 
the [DMRGObserver](@ref) documentation page.

## Defining a Custom Observer

To define a custom observer, just make a struct with
any name and internal fields you would like, and make
this struct a subtype of `AbstractObserver`.

For example, let's make a type called `DemoObserver`
as:

```julia
mutable struct DemoObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

```

In this minimal example, our `DemoObserver` 
contains a field `energy_tol` which we can use to set
an early-stopping condition for DMRG, and an field
`last_energy` which our observer will use internally
to keep track of changes to the energy after each sweep.

Now to give our `DemoObserver` type a useful behavior
we need to define overloads of the methods `measure!`
and `checkdone!`. 

### Overloading the `checkdone!` method

Let's start with the `checkdone!` method. After
each sweep of DMRG, the `checkdone!` method is 
passed the observer object, as well as a set of keyword
arguments which currently include:
  - energy: the current energy
  - psi: the current wavefunction MPS
  - sweep: the number of the sweep that just finished
  - outputlevel: an integer stating the desired level of output

If the `checkdone!` function returns `true`, then the DMRG
routine stops (recall that `checkdone!` is called only at the 
end of a sweep).

In our example, we will just compare the `energy` keyword
argument to the `last_energy` variable held inside the `DemoObserver`:

```julia
function ITensors.checkdone!(o::DemoObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end
```

(Recall that in order to properly overload the default behavior,
the `checkdone!` method has to be imported from the ITensors module
or preceded with `ITensors.`)


### Overloading the `measure!` method

The other method that an observer can overload is `measure!`.
This method is called at every step of DMRG, so at every 
site and for every sweep. The `measure!` method is passed
the current observer object and a set of keyword arguments
which include:
   - energy: the energy after the current step of DMRG
   - psi: the current wavefunction MPS 
   - bond: the bond `b` that was just optimized, corresponding to sites `(b,b+1)` in the two-site DMRG algorithm
   - sweep: the current sweep number
   - sweep\_is\_done: true if at the end of the current sweep, otherwise false
   - half_sweep: the half-sweep number, equal to 1 for a left-to-right, first half sweep, or 2 for the second, right-to-left half sweep
   - spec: the Spectrum object returned from factorizing the local superblock wavefunction tensor in two-site DMRG
   - outputlevel: an integer specifying the amount of output to show
   - projected_operator: projection of the linear operator into the current MPS basis

For our minimal `DemoObserver` example here, we will just make a `measure!` function
that prints out some of the information above, but in a more realistic setting one 
could use the MPS `psi` to perform essentially arbitrary measurements.

```julia
function ITensors.measure!(o::DemoObserver; kwargs...)
  energy = kwargs[:energy]
  sweep = kwargs[:sweep]
  bond = kwargs[:bond]
  outputlevel = kwargs[:outputlevel]

  if outputlevel > 0
    println("Sweep $sweep at bond $bond, the energy is $energy")
  end
end
```

## Calling DMRG with the Custom Observer

After defining an observer type and overloading at least one of the 
methods `checkdone!` or `measure!` for it, one can construct an
object of this type and pass it to the ITensor [`dmrg`](@ref) function
using the `observer` keyword argument.

Continuing with our `DemoObserver` example above:

```julia
obs = DemoObserver(1E-4) # use an energy tolerance of 1E-4
energy, psi = dmrg(H,psi0,sweeps; observer=obs, outputlevel=1)
```

## Complete Sample Code

```julia
using ITensors

mutable struct DemoObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

function ITensors.checkdone!(o::DemoObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end

function ITensors.measure!(o::DemoObserver; kwargs...)
  energy = kwargs[:energy]
  sweep = kwargs[:sweep]
  bond = kwargs[:bond]
  outputlevel = kwargs[:outputlevel]

  if outputlevel > 0
    println("Sweep $sweep at bond $bond, the energy is $energy")
  end
end

let
  N = 10
  etol = 1E-4

  s = siteinds("S=1/2",N)

  a = OpSum()
  for n=1:N-1
    a += "Sz",n,"Sz",n+1
    a += 0.5,"S+",n,"S-",n+1
    a += 0.5,"S-",n,"S+",n+1
  end
  H = MPO(a,s)
  psi0 = randomMPS(s,4)

  nsweeps = 5
  cutoff = 1E-8
  maxdim = [10,20,100]

  obs = DemoObserver(etol)

  println("Starting DMRG")
  energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=1)

  return
end
```
