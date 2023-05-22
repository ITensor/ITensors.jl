# DMRG Code Examples

## Perform a basic DMRG calculation 

Because tensor indices in ITensor have unique identities, before we can make a Hamiltonian
or a wavefunction we need to construct a "site set" which will hold the site indices defining
the physical Hilbert space:

```julia
N = 100
sites = siteinds("S=1",N)
```

Here we have chosen to create a Hilbert space of N spin 1 sites. The string "S=1"
denotes a special Index tag which hooks into a system that knows "S=1" indices have
a dimension of 3 and how to create common physics operators like "Sz" for them.

Next we'll make our Hamiltonian matrix product operator (MPO). A very 
convenient way to do this is to use the OpSum helper type which lets 
us input a Hamiltonian (or any sum of local operators) in similar notation
to pencil-and-paper notation:

```julia
os = OpSum()
for j=1:N-1
  os += 0.5,"S+",j,"S-",j+1
  os += 0.5,"S-",j,"S+",j+1
  os += "Sz",j,"Sz",j+1
end
H = MPO(os,sites)
```

In the last line above we convert the OpSum helper object to an actual MPO.

Before beginning the calculation, we need to specify how many DMRG sweeps to do and
what schedule we would like for the parameters controlling the accuracy.
These parameters can be specified as follows:

```julia
nsweeps = 5 # number of sweeps is 5
maxdim = [10,20,100,100,200] # gradually increase states kept
cutoff = [1E-10] # desired truncation error
```

The random starting wavefunction `psi0` must be defined in the same Hilbert space
as the Hamiltonian, so we construct it using the same collection of site indices:

```julia
psi0 = randomMPS(sites,2)
```

Here we have made a random MPS of bond dimension 2. We could have used a random product
state instead, but choosing a slightly larger bond dimension can help DMRG avoid getting
stuck in local minima. We could also set psi to some specific initial state using the 
`MPS` constructor, which is actually required if we were conserving QNs.

Finally, we are ready to call DMRG:

```julia
energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
```

When the algorithm is done, it returns the ground state energy as the variable `energy` and an MPS 
approximation to the ground state as the variable `psi`.

Below you can find a complete working code that includes all of these steps:

```julia
using ITensors

let
  N = 100
  sites = siteinds("S=1",N)

  os = OpSum()
  for j=1:N-1
    os += 0.5,"S+",j,"S-",j+1
    os += 0.5,"S-",j,"S+",j+1
    os += "Sz",j,"Sz",j+1
  end
  H = MPO(os,sites)

  nsweeps = 5 # number of sweeps is 5
  maxdim = [10,20,100,100,200] # gradually increase states kept
  cutoff = [1E-10] # desired truncation error

  psi0 = randomMPS(sites,2)

  energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

  return
end
```

## Using a Custom Observer for DMRG

An Observer is any object which can be used to perform custom measurements throughout
a DMRG calculation and to stop a DMRG calculation early. Because an Observer has
access to the entire wavefunction at every step, a wide range of customization is
possible.

For detailed examples of making custom Observers, see the [Observer](@ref observer)
section of the documentation.


## DMRG Calculation with Mixed Local Hilbert Space Types

The following fully-working example shows how to set up a calculation
mixing S=1/2 and S=1 spins on every other site of a 1D system. The 
Hamiltonian involves Heisenberg spin interactions with adjustable
couplings between sites of the same spin or different spin.

Note that the only difference from a regular ITensor DMRG calculation
is that the `sites` array has Index objects which alternate in dimension
and in which physical tag type they carry, whether `"S=1/2"` or `"S=1"`.
(Try printing out the sites array to see!)
These tags tell the OpSum system which local operators to use for these
sites when building the Hamiltonian MPO.

```julia
using ITensors

let
  N = 100

  # Make an array of N Index objects with alternating
  # "S=1/2" and "S=1" tags on odd versus even sites
  # (The first argument n->isodd(n) ... is an 
  # on-the-fly function mapping integers to strings)
  sites = siteinds(n->isodd(n) ? "S=1/2" : "S=1",N)

  # Couplings between spin-half and
  # spin-one sites:
  Jho = 1.0 # half-one coupling
  Jhh = 0.5 # half-half coupling
  Joo = 0.5 # one-one coupling

  os = OpSum()
  for j=1:N-1
    os += 0.5*Jho,"S+",j,"S-",j+1
    os += 0.5*Jho,"S-",j,"S+",j+1
    os += Jho,"Sz",j,"Sz",j+1
  end
  for j=1:2:N-2
    os += 0.5*Jhh,"S+",j,"S-",j+2
    os += 0.5*Jhh,"S-",j,"S+",j+2
    os += Jhh,"Sz",j,"Sz",j+2
  end
  for j=2:2:N-2
    os += 0.5*Joo,"S+",j,"S-",j+2
    os += 0.5*Joo,"S-",j,"S+",j+2
    os += Joo,"Sz",j,"Sz",j+2
  end
  H = MPO(os,sites)

  nsweeps = 10
  maxdim = [10,10,20,40,80,100,140,180,200]
  cutoff = [1E-8]

  psi0 = randomMPS(sites,4)

  energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

  return
end
```

## Use a Sum of MPOs in DMRG

One version of the ITensor `dmrg` function accepts an array of MPOs
`[H1,H2,H3]` (or any number of MPOs you want). This version of DMRG
will find the ground state of `H1+H2+H3`. Internally it does not
actually sum these MPOs, but loops over them during each step of 
the "eigensolver" at the core of the DMRG algorithm, so it is usually
more efficient than if the MPOs had been summed together into a single MPO.

To use this version of DMRG, say you have MPOs `H1`, `H2`, and `H3`. 
Then call DMRG like this:
```julia
energy,psi = dmrg([H1,H2,H3],psi0; nsweeps, maxdim, cutoff)
```

## Make a 2D Hamiltonian for DMRG

You can use the OpSum system to make 2D Hamiltonians
much in the same way you make 1D Hamiltonians: by looping over
all of the bonds and adding the interactions on these bonds to
the OpSum. 

To help with the logic of 2D lattices, ITensor pre-defines
some helper functions which
return an array of bonds. Each bond object has an
"s1" field and an "s2" field which are the integers numbering
the two sites the bond connects.
(You can view the source for these functions at [this link](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/lattices.jl).)

The two provided functions currently are `square_lattice` and 
`triangular_lattice`. It is not hard to write your own similar lattice
functions as all they have to do is define an array of `ITensors.LatticeBond`
structs or even a custom struct type you wish to define. We welcome any
user contributions of other lattices that ITensor does not currently offer.

Each lattice function takes an optional named argument
"yperiodic" which lets you request that the lattice should
have periodic boundary conditions around the y direction, making
the geometry a cylinder.

**Full example code:**

```julia
using ITensors

let
  Ny = 6
  Nx = 12

  N = Nx*Ny

  sites = siteinds("S=1/2", N;
                   conserve_qns = true)

  # Obtain an array of LatticeBond structs
  # which define nearest-neighbor site pairs
  # on the 2D square lattice (wrapped on a cylinder)
  lattice = square_lattice(Nx, Ny; yperiodic = false)

  # Define the Heisenberg spin Hamiltonian on this lattice
  os = OpSum()
  for b in lattice
    os .+= 0.5, "S+", b.s1, "S-", b.s2
    os .+= 0.5, "S-", b.s1, "S+", b.s2
    os .+=      "Sz", b.s1, "Sz", b.s2
  end
  H = MPO(os,sites)

  state = [isodd(n) ? "Up" : "Dn" for n=1:N]
  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  psi0 = randomMPS(sites,state,20)

  nsweeps = 10
  maxdim = [20,60,100,100,200,400,800]
  cutoff = [1E-8]

  energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

  return
end
```

## Compute excited states with DMRG 

ITensor DMRG accepts additional MPS wavefunctions as a optional, extra argument.
These additional 'penalty states' are provided as an array of MPS just 
after the Hamiltonian, like this:

```julia
energy,psi3 = dmrg(H,[psi0,psi1,psi2],psi3_init; nsweeps, maxdim, cutoff)
```

Here the penalty states are `[psi0,psi1,psi2]`. 
When these are provided, the DMRG code minimizes the
energy of the current MPS while also reducing its overlap 
(inner product) with the previously provided MPS. If these overlaps become sufficiently small,
then the computed MPS is an excited state. So by finding the ground
state, then providing it to DMRG as a "penalty state" or previous state
one can compute the first excited state. Then providing both of these, one can
get the second excited state, etc.

A  keyword argument called `weight` can also be provided to
the `dmrg` function when penalizing overlaps to previous states. The 
`weight` parameter is multiplied by the overlap with the previous states,
so sets the size of the penalty. It should be chosen at least as large
as the (estimated) gap between the ground and first excited states.
Otherwise the optimal value of the weight parameter is not so obvious,
and it is best to try various weights during initial test calculations.

Note that when the system has conserved quantum numbers, a superior way
to find excited states can be to find ground states of quantum number (or symmetry)
sectors other than the one containing the absolute ground state. In that
context, the penalty method used below is a way to find higher excited states
within the same quantum number sector.
  
**Full Example code:**

```julia
using ITensors

let
  N = 20

  sites = siteinds("S=1/2",N)

  h = 4.0
  
  weight = 20*h # use a large weight
                # since gap is expected to be large


  #
  # Use the OpSum feature to create the
  # transverse field Ising model
  #
  # Factors of 4 and 2 are to rescale
  # spin operators into Pauli matrices
  #
  os = OpSum()
  for j=1:N-1
    os += -4,"Sz",j,"Sz",j+1
  end
  for j=1:N
    os += -2*h,"Sx",j;
  end
  H = MPO(os,sites)


  #
  # Make sure to do lots of sweeps
  # when finding excited states
  #
  nsweeps = 30
  maxdim = [10,10,10,20,20,40,80,100,200,200]
  cutoff = [1E-8]
  noise = [1E-6]

  #
  # Compute the ground state psi0
  #
  psi0_init = randomMPS(sites,linkdims=2)
  energy0,psi0 = dmrg(H,psi0_init; nsweeps, maxdim, cutoff, noise)

  println()

  #
  # Compute the first excited state psi1
  #
  psi1_init = randomMPS(sites,linkdims=2)
  energy1,psi1 = dmrg(H,[psi0],psi1_init; nsweeps, maxdim, cutoff, noise, weight)

  # Check psi1 is orthogonal to psi0
  @show inner(psi1,psi0)


  #
  # The expected gap of the transverse field Ising
  # model is given by Eg = 2*|h-1|
  #
  # (The DMRG gap will have finite-size corrections)
  #
  println("DMRG energy gap = ",energy1-energy0);
  println("Theoretical gap = ",2*abs(h-1));

  println()

  #
  # Compute the second excited state psi2
  #
  psi2_init = randomMPS(sites,linkdims=2)
  energy2,psi2 = dmrg(H,[psi0,psi1],psi2_init; nsweeps, maxdim, cutoff, noise, weight)

  # Check psi2 is orthogonal to psi0 and psi1
  @show inner(psi2,psi0)
  @show inner(psi2,psi1)

  return
end
```

## Printing the Entanglement Entropy at Each Step

To obtain the entanglement entropy of an MPS at each step during a DMRG calculation,
you can use the [Observer](@ref observer) system to make a custom observer object that prints out
this information.

First we define our custom observer type, `EntanglementObserver`, and overload the `measure!` function
for it:

```julia
mutable struct EntanglementObserver <: AbstractObserver
end

function ITensors.measure!(o::EntanglementObserver; bond, psi, half_sweep, kwargs...)
  wf_center, other = half_sweep==1 ? (psi[bond+1],psi[bond]) : (psi[bond],psi[bond+1])
  U,S,V = svd(wf_center, uniqueinds(wf_center,other))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  println("  Entanglement across bond $bond = $SvN")
end
```

The `measure!` function grabs certain helpful keywords passed to it by DMRG, such as what bond DMRG 
has just finished optimizing.

Here is a complete sample code including constructing the observer and passing it to DMRG:

```julia
using ITensors

mutable struct EntanglementObserver <: AbstractObserver
end

function ITensors.measure!(o::EntanglementObserver; bond, psi, half_sweep, kwargs...)
  wf_center, other = half_sweep==1 ? (psi[bond+1],psi[bond]) : (psi[bond],psi[bond+1])
  U,S,V = svd(wf_center, uniqueinds(wf_center,other))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  println("  Entanglement across bond $bond = $SvN")
end

let
  N = 100

  s = siteinds("S=1/2",N)

  a = OpSum()
  for n=1:N-1
    a += "Sz",n,"Sz",n+1
    a += 0.5,"S+",n,"S-",n+1
    a += 0.5,"S-",n,"S+",n+1
  end
  H = MPO(a,s)
  psi0 = randomMPS(s,linkdims=4)

  nsweeps = 5
  maxdim = [10,20,80,160]
  cutoff = 1E-8

  observer = EntanglementObserver()

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, observer, outputlevel=2)

  return
end
```

Example output:
```
...
Sweep 2, half 2, bond (35,36) energy=-44.08644657103751
  Truncated using cutoff=1.0E-08 maxdim=20 mindim=1
  Trunc. err=2.54E-07, bond dimension 20
  Entanglement across bond 35 = 0.7775882479059774
Sweep 2, half 2, bond (34,35) energy=-44.086696891668424
  Truncated using cutoff=1.0E-08 maxdim=20 mindim=1
  Trunc. err=2.12E-07, bond dimension 20
  Entanglement across bond 34 = 0.7103532704635472
Sweep 2, half 2, bond (33,34) energy=-44.08696190368391
  Truncated using cutoff=1.0E-08 maxdim=20 mindim=1
  Trunc. err=1.29E-07, bond dimension 20
  Entanglement across bond 33 = 0.7798362911744212
...
```

If you only want to see the maximum entanglement during each sweep, you can add a field to the EntanglementObserver
object that saves the maximum value encountered so far and keep overwriting this field, printing out the most recently observed maximum at the end of each sweep.


## Monitoring the Memory Usage of DMRG

To monitor how much memory (RAM) a DMRG calculation is using while it is running,
you can use the [Observer](@ref observer) system to make a custom observer object that prints out
this information. Also the `Base.summarysize` function, which returns the size 
in bytes of any Julia object is very helpful here.

First we define our custom observer type, `SizeObserver`, and overload the `measure!` function
for it:

```julia
mutable struct SizeObserver <: AbstractObserver
end

function ITensors.measure!(o::SizeObserver; bond, half_sweep, psi, projected_operator, kwargs...)
  if bond==1 && half_sweep==2
    psi_size =  Base.format_bytes(Base.summarysize(psi))
    PH_size =  Base.format_bytes(Base.summarysize(projected_operator))
    println("|psi| = $psi_size, |PH| = $PH_size")
  end
end
```

The `measure!` function grabs certain helpful keywords passed to it by DMRG, checking 
`if bond==1 && half_sweep==2` so that it only runs when at the end of a full sweep.

When it runs, it calls `Base.summarysize` on the wavefunction `psi` object and the `projected_operator` object. The `projected_operator`, which is the matrix (Hamiltonian) wrapped into the current MPS basis, is usually the largest-sized object in a DMRG calculation. The code also uses `Base.format_bytes` to turn an integer representing bytes into a human-readable string.

Here is a complete sample code including constructing the observer and passing it to DMRG:

```julia
using ITensors

mutable struct SizeObserver <: AbstractObserver
end

function ITensors.measure!(o::SizeObserver; bond, sweep, half_sweep, psi, projected_operator, kwargs...)
  if bond==1 && half_sweep==2
    psi_size =  Base.format_bytes(Base.summarysize(psi))
    PH_size =  Base.format_bytes(Base.summarysize(projected_operator))
    println("After sweep $sweep, |psi| = $psi_size, |PH| = $PH_size")
  end
end

let
  N = 100

  s = siteinds("S=1/2",N)

  a = OpSum()
  for n=1:N-1
    a += "Sz",n,"Sz",n+1
    a += 0.5,"S+",n,"S-",n+1
    a += 0.5,"S-",n,"S+",n+1
  end
  H = MPO(a,s)
  psi0 = randomMPS(s,linkdims=4)

  nsweeps = 5
  maxdim = [10,20,80,160]
  cutoff = 1E-8

  obs = SizeObserver()

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, observer=obs)

  return
end
```

Example output:
```
After sweep 1, |psi| = 211.312 KiB, |PH| = 593.984 KiB
After sweep 1 energy=-43.95323393592883  maxlinkdim=10 maxerr=8.26E-06 time=0.098
After sweep 2, |psi| = 641.000 KiB, |PH| = 1.632 MiB
After sweep 2 energy=-44.10791340895817  maxlinkdim=20 maxerr=7.39E-07 time=0.132
After sweep 3, |psi| = 1.980 MiB, |PH| = 5.066 MiB
After sweep 3 energy=-44.12593605906466  maxlinkdim=44 maxerr=9.96E-09 time=0.256
After sweep 4, |psi| = 2.863 MiB, |PH| = 7.246 MiB
After sweep 4 energy=-44.127710946536645  maxlinkdim=56 maxerr=9.99E-09 time=0.445
After sweep 5, |psi| = 3.108 MiB, |PH| = 7.845 MiB
After sweep 5 energy=-44.127736798226536  maxlinkdim=57 maxerr=9.98E-09 time=0.564
```
