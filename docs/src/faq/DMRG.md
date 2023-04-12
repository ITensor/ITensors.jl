# Density Matrix Renormalization Group (DMRG) Frequently Asked Questions

## Ensuring a DMRG calculation is converged

While DMRG calculations can be extremely quick to converge in the best cases,
convergence can be slower for cases such as gapless systems or quasi-two-dimensional systems.
So it becomes important to know if a DMRG calculation is converged i.e. has been run
long enough with enough resources (large enough MPS bond dimension).

Unfortunately **there is no automatic or bulletproof check for DMRG convergence**.
However, there are a number of reliable heuristics you can use to check convergence.
We list some of these with the most fundamental and important ones first:

* Run your DMRG calculation on a **smaller system** and compare with another method, such
  as an exact diagonalization. If the agreement is good, then gradually try larger
  systems and see if the physical properties are roughly consistent and similar (i.e.
  the density profile has similar features).

* Make sure to check a **wide range of properties** - not just the energy. See if these
  look plausible by plotting and visually inspecting them. For example: if your system has 
  left-right reflection symmetry, does the density or magnetization also have this symmetry? 
  If the ground  state of your system is expected to have a total ``S^z`` of zero, does your 
  ground state have this property?

* Make sure to run your DMRG calculation for **different numbers of sweeps** to see if 
  the results change. For example, if you run DMRG for 5 sweeps but are unsure of convergence,
  try running it for 10 sweeps: is the energy the same or has it significantly decreased?
  If 10 sweeps made a difference, try 20 sweeps.

* Try setting the `eigsolve_krylovdim` keyword argument to a higher value (the default is 3).
  This may make slowly-converging calculations converge in fewer sweeps, but setting it 
  too high can make each sweep run slowly.

* Inspect the the **DMRG output**. 
  The ITensor DMRG code reports the maximum bond or link dimension and maximum truncation error
  after each sweep. (The maximums here mean over each DMRG substep making up one sweep.)
  Is the maximum dimension or "maxlinkdim" reported by the DMRG output quickly reaching 
  and saturating the maxdim value you set for each sweep? Is the maximum truncation error 
  "maxerr" consistently reaching large values, larger than 1E-5? 
  Then it you may need to raise the maxdim parameter for your later sweeps, 
  so that DMRG is allowed to use a larger bond dimension and thus reach a better accuracy.

* Compute the **energy variance** of an MPS to check whether it is an eigenstate. To do this
  in ITensor, you can use the following code where `H` is your Hamiltonian MPO
  and `psi` is the wavefunction you want to check:

  ```julia
  H2 = inner(H,psi,H,psi)
  E = inner(psi',H,psi)
  var = H2-E^2
  @show var
  ```
  
  Here `var` is the quantity ``\langle H^2 \rangle - \langle H \rangle^2``.
  The closer `var` is to zero, the more precisely `psi` is an eigenstate of `H`. Note
  that this check does not ensure that `psi` is the ground state, but only one of the 
  eigenstates.


## Preventing DMRG from getting stuck in a local minimum

While DMRG has very robust convergence properties when the initial MPS is close to the global
minimum, if it is far from the global minumum then there is _no guarantee_ that DMRG will
be able to find the true ground state. This problem is exacerbated for quantum number conserving
DMRG where the search space is more constrained.

Thus it is very important to perform a number of checks to ensure that the result you
get from DMRG is actually converged. To learn about these checks, see the previous question.

When DMRG is failing to converge, here are some of the steps you can take to improve things:

* _The most important and useful technique_ is to turn on the **noise term** feature of DMRG.
  To do this, just set the `noise` parameter of each sweep to a small, non-zero value, making
  this value very small (1E-11, say) or zero by the last sweep. (Experiment with different
  values on small systems to see which noise magnitudes help.) Here is an example of 
  defining DMRG accuracy or sweep parameters with a non-zero noise set for the first three sweeps:

  ```julia
  nsweeps = 10
  maxdim = [100, 200, 400, 800, 1600]
  cutoff = [1E-6]
  noise = [1E-6, 1E-7, 1E-8, 0.0]
  ...
  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, noise)
  ```

* Try using a initial MPS with properties close to the ground state you are looking for.
  For example, the ground state of a system of electrons typically has a density which is
  spread out over the whole system. So if your initial state has all of the electrons bunched
  up on the left-hand side only, it can take DMRG a very long time to converge.

* Try using a random MPS with a modestly large bond dimension. ITensor offers a function
  called [`randomMPS`](@ref) which can be used to make random MPS in both the quantum number (QN)
  conserving and non-QN conserving cases. Because random MPS have properties
  which are "typical" of most ground states, they can be good initial states for DMRG.

* Try DMRG on a closely related Hamiltonian for which convergence is easier to obtain
  (be creative here: it could be your Hamiltonian with interactions turned off, or
   with interactions only within, but not between, small local patches). Take the
  output of this first calculation and use it as input for DMRG with the full Hamiltonian.

* In stubborn cases, try other methods for finding the ground state which are slower, but
  have a better chance of succeeding. A key example is imaginary time evolution, which 
  always reaches the ground state if (a) performed accurately on (b) a state which is 
  not orthogonal to the ground state. After doing some amount of imaginary time evolution,
  use the resulting MPS as an initial state for DMRG obtain a higher-accuracy solution.

## How to do periodic boundary condition DMRG

The short answer to how to do fully periodic boundary condition DMRG in ITensor is that
you simply input a **periodic Hamiltonian** into our OpSum system and make the MPO
form of your Hamiltonian in the usual way. For example, for a chain of N sites with nearest-neighbor
interactions, you include a term that connects site 1 to site N. For a one-dimensional Ising model 
chain Hamiltonian this would look like:

```
sites = siteinds("S=1/2",N)

hterms = OpSum()
for j=1:(N-1)
  hterms += "Sz",j,"Sz",j+1
end
hterms += "Sz",1,"Sz",N  # term 'wrapping' around the ring

H = MPO(hterms,sites)
```

For two-dimensional DMRG calculations, where the most common approach is to use 
periodic boundary conditions in the y-direction only, and not in the x-direction, 
you do a similar step in making your OpSum input to ITensor DMRG: you include 
terms wrapping around the periodic cylinder in the y direction but not in the x direction.

However, fully periodic boundary conditions are only recommended for small systems 
when absolutely needed, and in general are not recommended. For a longer discussion 
of alternatives to using fully periodic boundaries, see the next section below.

The reason fully periodic boundary conditions (periodic in x in 1D, and periodic in both x 
and y in 2D) are not recommended in general is that the DMRG algorithm, as we are defining it
here, optimizes an **open-boundary MPS**. So if you input a periodic-boundary Hamiltonian, there
is a kind of "mismatch" that happens where you can still get the correct answer, but it 
requires much more resources (a larger bond dimension and more sweeps) to get good 
accuracy. There has been some research into "truly" periodic DMRG, [^Pippan] that is DMRG that
optimizes an MPS with a ring-like topology, but it is not widely used, is still an
open area of algorithm development, and is not currently available in ITensor.


## What boundary conditions should I choose: open, periodic, or infinite?

One of the weaknesses of the density matrix renormalization group (DMRG), and its time-dependent or finite-temperature extensions, is that it works poorly with periodic boundary conditions. This stems from the fact that conventional DMRG optimizes over open-boundary matrix product state (MPS) wavefunctions whether or not the Hamiltonian includes periodic interactions.

But this begs the question, when are periodic boundary conditions (PBC) really needed? DMRG offers
some compelling alternatives to PBC:

* Use open boundary conditions (OBC). Though this introduces edge effects, the number of states needed
  to reach a given accuracy is _significantly_ lower than with PBC (see next section below). 
  For gapped systems DMRG scales linearly with system size, meaning often one can study systems with many hundreds or even thousands of sites. Last but not least, open boundaries are often more natural. For studying systems which spontaneously break symmetry, adding "pinning" fields on the edge is often a very nice way to tip the balance toward a certain symmetry broken state while leaving the bulk unmodified.

* Use smooth boundary conditions. The basic idea is to use OBC but 
  send the Hamiltonian parameters smoothly to zero at the boundary so that the system can not "feel"
  the boundary. For certain systems this can significantly reduce edge effects.[^Smooth1][^Smooth2][^Smooth3]

[^Smooth1]: [Smooth boundary conditions for quantum lattice systems](http://dx.doi.org/10.1103/PhysRevLett.71.4283), M. Vekic and Steven R. White, _Phys. Rev. Lett._ **71**, [4283](http://dx.doi.org/10.1103/PhysRevLett.71.4283) (1993) cond-mat/[9310053](http://arxiv.org/abs/cond-mat/9310053)

[^Smooth2]: [Hubbard model with smooth boundary conditions](http://dx.doi.org/10.1103/PhysRevB.53.14552), M. Vekic and Steven R. White, _Phys. Rev. B_ **53**, [14552](http://dx.doi.org/10.1103/PhysRevB.53.14552) (1996) cond-mat/[9601009](http://arxiv.org/abs/cond-mat/9601009)

[^Smooth3]: [Grand canonical finite-size numerical approaches: A route to measuring bulk properties in an applied field](http://link.aps.org/doi/10.1103/PhysRevB.86.041108), Chisa Hotta and Naokazu Shibata, _Phys. Rev. B_ **86**, [041108](http://link.aps.org/doi/10.1103/PhysRevB.86.041108) (2012) 
 


* Use "infinite boundary conditions", that is, use infinite DMRG in the form of an algorithm like iDMRG or VUMPS. This has a cost that can be even less than with OBC yet is completely free of finite-size effects.

However, there are a handful of cases where PBC remains preferable despite the extra overhead. A few such cases are:

* Benchmarking DMRG against another code that uses PBC, such as a Monte Carlo or exact diagonalization code.

* Extracting the central charge of a critical one-dimensional system described by a CFT. In practice, using PBC can give an accurate central charge even for quite small systems by fitting the subsystem entanglement entropy to the CFT scaling form.

* Checking for the presence or absence of topological effects. These could be edge effects (the Haldane
  phase has a four-fold ground state degeneracy with OBC, but not with PBC), or could be related to some global topological sector that is ill-defined with PBC (e.g. periodic vs. antiperiodic boundary conditions for the transverse field Ising model).

(Note that in the remaining discussion, by PBC I mean  *fully periodic* boundary conditions in all directions.
For the case of DMRG applied to quasi-two-dimensional systems, it remains a good practice to use
periodic boundaries in the shorter direction, while still using open (or infinite) boundaries
in the longer direction along the DMRG/MPS path.)

Below I discuss more about the problems with using PBC, as well as some misconceptions about when PBC seems necessary even though there are better alternatives.

#### Drawbacks of Periodic Boundary Conditions

Periodic boundary conditions are straightforward to implement in conventional DMRG. The simplest approach is to include a "long bond" directly connecting site 1 to site N in the Hamiltonian. However this 
naive approach has a major drawback: if open-boundary DMRG achieves a given accuracy when keeping ``m`` states (bond dimension of size ``m``), then to reach the same accuracy with PBC one must keep closer to ``m^2`` states! The reason is that now every bond of the MPS not only carries local entanglement as with OBC, but also the entanglement between the first and last sites. (There is an alternative DMRG algorithm[^Pippan] for periodic systems which may have better scaling than the above approach but has not been widely applied and tested, as far as I am aware, especially for
 2D or critical systems .)

[^Pippan]: [Efficient matrix-product state method for periodic boundary conditions](http://link.aps.org/doi/10.1103/PhysRevB.81.081103), P. Pippan, Steven R. White, and H.G. Evertz, _Phys. Rev. B_ **81**, [081103](http://link.aps.org/doi/10.1103/PhysRevB.81.081103)

The change in scaling from ``m`` to ``m^2``  is a severe problem.
For example, many gapped one-dimensional systems only require about ``m=100`` to reach good accuracy
(truncation errors of less than 1E-9 or so). To reach the same accuracy with naive PBC would then
require using 10,000 states, which can easily fill the RAM of a typical desktop computer for a large enough system, not to mention the extra time needed to work with larger matrices.

But poor scaling is not the only drawback of PBC. Systems that exhibit spontaneous symmetry breaking 
are simple to work with under OBC, where one has the additional freedom of applying edge pinning terms 
to drive the bulk into a specific symmetry sector. Using edge pinning reduces the bulk entanglement and makes measuring order parameters straightforward. Similarly one can use infinite DMRG to directly observe symmetry breaking effects.

But under PBC, order parameters remain equal to zero and can only be accessed through correlation functions. Though using correlation functions is often presented as the "standard" or "correct" approach, such reasoning pre-supposes that PBC is the best choice. Recent work in the quantum Monte Carlo community demonstrates that open boundaries with pinning fields can actually be a superior approach.[^Assaad]

[^Assaad]: [Pinning the Order: The Nature of Quantum Criticality in the Hubbard Model on Honeycomb Lattice](http://dx.doi.org/10.1103/PhysRevX.3.031010), Fakher F. Assaad and Igor F. Herbut, _Phys. Rev. X_ **3**, [031010](http://dx.doi.org/10.1103/PhysRevX.3.031010)


#### Cases Where Periodic BC Seems Necessary, But Open/Infinite BC Can be Better

Below are some cases where periodic boundary conditions seem to be necessary at a first glance. 
But in many of these cases, not only can open or infinite boundaries be just as successful, they 
can even be the better choice.

* _Measuring asymptotic properties of correlation functions_: much of our understanding of gapless one-dimensional systems comes from field-theoretic approaches which make specific predictions about asymptotic decays of various correlators. To test these predictions numerically, one must  work with large, translationally invariant systems with minimal edge effects. Using fully periodic boundary conditions satisfies these criteria. However, a superior choice is to use infinite DMRG, which combines the much better scaling of open-boundary DMRG with the ability to  measure correlators at _arbitrarily long_ distances by repeating the unit cell of the MPS wavefunction. Although truncating to a finite number of states imposes an effective correlation length on the system, this correlation length can reach many thousands of sites for quite moderate MPS bond dimensions. Karrasch and Moore took advantage of this fact to convincingly check the predictions of Luttinger liquid theory for one-dimensional systems of gapless fermions.[^Karrasch]

[^Karrasch]: [Luttinger liquid physics from the infinite-system density matrix renormalization group](http://dx.doi.org/10.1103/PhysRevB.86.155156), C. Karrasch and J.E. Moore, _Phys. Rev. B_ **86**, [155156](http://dx.doi.org/10.1103/PhysRevB.86.155156)

* _Studying two-dimensional topological order_: a hallmark of intrinsic topological order is the presence of a robust ground state degeneracy when the system is put on a torus. Also many topological phases  have gapless edge states which can cause problems for numerical calculations. Thus one might think that fully periodic BC are the best choice for studying topological phases. However,  topological phases have the same ground-state degeneracy on an infinite cylinder as they do on a torus.[^Zhang]. Cincio and Vidal exploited this fact to use infinite DMRG to study a variety of topological phases [^Cincio]. One part of their calculation did actually require obtaining ground states on a torus, but they accomplished this by taking a finite segment of an infinite MPS  and connecting its ends. This approach does not give the true ground state of the torus but was sufficient  for their calculation and was arguably closer to the true two-dimensional physics.

[^Zhang]: [Quasiparticle statistics and braiding from ground-state entanglement](http://dx.doi.org/10.1103/PhysRevB.85.235151), Yi Zhang, Tarun Grover, Ari Turner, Masaki Oshkawa, and Ashvin Vishwanath, _Phys. Rev. B_ **85**, [235151](http://dx.doi.org/10.1103/PhysRevB.85.235151)

[^Cincio]: [Characterizing Topological Order by Studying the Ground States on an Infinite Cylinder](http://link.aps.org/doi/10.1103/PhysRevLett.110.067208), L. Cincio and G. Vidal, _Phys. Rev. Lett._ **110**, [067208](http://link.aps.org/doi/10.1103/PhysRevLett.110.067208)

* _Obtaining bulk gaps_: 
  DMRG has the ability to "target" low-lying excited states or to obtain such
  states by constraining them to be orthogonal to the ground state. However, with OBC, 
  localized excitations can get stuck to the edges and not reveal the true bulk gap behavior. 
  Thus one may conclude that PBC is necessary. But using open or infinite boundaries remains 
  the better choice because they allow much higher accuracy.

  To deal with the presence of edges in OBC, one can use "restricted sweeping". Here one sweeps across the full system to obtain the ground state. Then, to obtain the first excited state one only sweeps through the full system to obtain the ground state. Then, to obtain the first excited state one only sweeps through the near the edges. This traps the particle in a "soft box" which still lets its wavefunction mix with the basis that describes the ground state outside the restricted sweeping region.

  Within infinite DMRG, boundary effects are rigorously absent if the calculation has converged. To compute bulk gaps one again uses a type of restricted sweeping known in the literature as "infinite boundary conditions". For more see the work by Phien, Vidal, and McCulloch.[^Phien]

[^Phien]: [Infinite boundary conditions for matrix product state calculations](http://link.aps.org/doi/10.1103/PhysRevB.86.245107), Ho N. Phien, G. Vidal, and Ian P. McCulloch _Phys. Rev. B_ **86**, [245107](http://link.aps.org/doi/10.1103/PhysRevB.86.245107)


In conclusion, consider carefully whether you really need to use periodic boundary conditions, as they impose a steep computational cost within DMRG. Periodic BC can actually be worse for the very types of measurements where they are  often presented as the best or "standard" choice. Many of the issues periodic boundaries circumvent can be avoided more elegantly by using infinite DMRG, or when that is not applicable, by using open boundary conditions with sufficient care.

