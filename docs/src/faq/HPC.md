# High Performance Computing (HPC) Frequently Asked Questions

## My code is using a lot of RAM - what can I do about this?

Tensor network algorithms can often use a large amount of RAM. On top
of this essential fact, the Julia programming languge is "garbage collected"
which means that unused memory isn't given back to the operating system right away, 
but only when the Julia runtime dynamically reclaims it. When your code
allocates memory very rapidly, this can lead to high memory usage overall.

Fortunately there are various steps you can take to keep the memory usage of your code under control.

### 1. Avoid Repeatedly Allocating, Especially in Fast or "Hot" Loops

More memory gets used whenever your code "allocates", which happens most commonly
when you use dynamic storage types like `Vector` and `Matrix`. If you have a code
pattern where you allocate or resize an array or vector inside a 'hot' loop, 
meaning a loop that iterates quickly very many times, the memory from the previous 
allocations may pile up very quickly before the next garbage collector run. 

To avoid this, allocate the array once before the loop begins if possible,
then overwrite its contents during each iteration. More generally, try as much as
possible to estimate the sizes of dynamic resources ahead of time. Or do one allocation
that creates a large enough "workspace" that dynamic algorithms can reuse part of without
reallocating the whole workspace (i.e. making a large array once then using portions of it
when smaller arrays are needed).

### 2. Use the `--heap-size-hint` Flag

A simple step you can take to help with overall memory usage is to pass
the `--heap-size-hint` flag to the Julia program when you start it. For example,
you can call Julia as:
```
julia --heap-size-hint=60G
```
When you pass this heap size, Julia will try to keep the memory usage at or below this
value if possible.

In cases where this does not work, your code simply may be allocating too much memory.
Be sure not to allocate over and over again inside of "hot" loops which execute many times.

Another possibility is that you are simply working with a tensor network with large 
bond dimensions, which may fundamentally use a lot of memory. In those cases, you can 
try to use features such as "write to disk mode" of the ITensor DMRG code or other related
techniques. (See the `write_when_maxdim_exceeds` keyword of the ITensor `dmrg` function.)


### 3. In Rare Case, Force a Garbage Collection Run

In some rare cases, such as when your code cannot be optimized to avoid any more allocations
or when the `--heap-size-hint` provided above is not affecting the behavior of the Julia
garbage collector, you can force the garbage collector (GC) to run at a specific point
in your code by calling:
```
GC.gc()
```
Alternatively, you can call `GC.gc(true)` to force a "full run" rather than just collecting
a more 'young' subset of previous allocations.

While this approach works well to reduce memory usage, it can have the unfortunate downside
of slowing down your code each time the garbage collector runs, which can be especially
harmful to multithreaded or parallel algorithms. Therefore, if this approach must be used
try calling `GC.gc()` as infrequently as possible and ideally only in the outermost functions
and loops of your code (highest levels of your code).


## Can Julia Be Used to Perform Parallel, Distributed Calculations on Large Clusters?

Yes. The Julia ecosystem offers multiple approaches to parallel computing across multiple
machines including on large HPC clusters and including GPU resources. 

For an overall view of some of these options, the [Julia on HPC Clusters](https://juliahpc.github.io/JuliaOnHPCClusters/) website is a good resource.

Some of the leading approaches to parallelism in Julia are:
* MPI, through the [MPI.jl](https://juliaparallel.org/MPI.jl/latest/) package. Has the advantage of optionally using an MPI backend that is optimized for a particular cluster and possibly using fast interconnects like Infiniband.
* [Dagger](https://juliaparallel.org/Dagger.jl/dev/), a framework for parallel computing across all kinds of resources, like CPUs and GPUs, and across multiple threads and multiple servers.
* [Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/). Part of the base Julia library, giving tools to perform calculations distributed across multiple machines.


## Does My Cluster Admin Have to Install Julia for Me? What are the Best Practices for Installing Julia on Clusters?

The most common approach to installing and using Julia on clusters is for users to install their own Julia binary and dependencies, which is quite easy to do. However, for certain libraries like MPI.jl, there may be MPI backends that are preferred by the cluster administrator. Fortunately, it is possible for admins to set global defaults for such backends and other library preferences.

For more information on best practices for installing Julia on clusters, see the [Julia on HPC Clusters](https://juliahpc.github.io/JuliaOnHPCClusters/) website.




