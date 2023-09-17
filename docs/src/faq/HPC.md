# High Performance Computing (HPC) Frequently Asked Questions

## My code is using a lot of RAM - what can I do about this?

Tensor network algorithms can often use a large amount of RAM. However, on top
of this essential fact, the Julia programming languge is also "garbage collected"
which means that unused memory isn't given back to the operating system right away, 
but only on a schedule determined by the Julia runtime. In cases where you code
allocates a lot of memory very quickly, this can lead to high memory usage.

Fortunately, one simple step you can take to potentially help with this is to pass
the `--heap-size-hint` flag to the Julia program when you start it. For example,
you can call Julia as:
```
julia --heap-size-hint=100G
```
When you pass this heap size, Julia will try to keep the memory usage at or below this
value if possible.

In cases where this does not work, your code simply may be allocating too much memory.
Be sure not to allocate over and over again inside of "hot" loops which execute many times.

Another possibility is that you are simply working with a tensor network with large 
bond dimensions, which may fundamentally use a lot of memory. In those cases, you can 
try to use features such as "write to disk mode" of the ITensor DMRG code or other related
techniques. (See the `write_when_maxdim_exceeds` keyword of the ITensor `dmrg` function.)

