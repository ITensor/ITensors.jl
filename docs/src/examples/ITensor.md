# ITensor Code Examples


## Print Indices of an ITensor

Sometimes the printout of an ITensor can be rather large, whereas you
might only want to see its indices. For these cases, just wrap the
ITensor in the function `inds` like this:

```julia
@show inds(T)
```

or this

```julia
println("T inds = ",inds(T))
```

## Getting and Setting Elements of an ITensor

Say we have an ITensor constructed as:

```julia
i = Index(3,"index_i")
j = Index(2,"index_j")
k = Index(4,"index_k")

T = ITensor(i,j,k)
```

An ITensor constructed this way starts with all of its elements
equal to zero. (Technically it allocates no storage at all but this is
an implementation detail.)

**Setting Elements**

To set an element of this ITensor, such as the element where `(i,j,k) = (2,1,3)`,
you can do the following:

```julia
T[i=>2,j=>1,k=>3] = -3.2
```

In the Julia language, the notation `a=>b` is a built-in notation for making a `Pair(a,b)`
object.

Because the Index objects are passed to `T` along with their values, passing them in a different order has exactly the same effect:

```julia
# Both of these lines of code do the same thing:
T[j=>1,i=>2,k=>3] = -3.2
T[j=>1,k=>3,i=>2] = -3.2
```

**Getting Elements**

You can retrieve individual elements of an ITensor by accessing them through the same notation used to set elements:

```julia
el = T[j=>1,i=>2,k=>3]
println("The (i,j,k) = (2,1,3) element of T is ",el)
```

## Arithmetic With ITensors

ITensors can be added and subtracted and multiplied by scalars just like plain tensors can. But ITensors have the additional feature that you can add and subtract them even if their indices are in a different order from each other, as long as they have the same collection of indices.

For example, say we have ITensors `A`, `B`, and `C`:
```julia
i = Index(3,"i")
j = Index(2,"j")
k = Index(4,"k")

A = randomITensor(i,j,k)
B = randomITensor(i,j,k)
C = randomITensor(k,i,j)
```
Above we have initialized these ITensors to have random elements, just for the sake of this example.

We can then add or subtract these ITensors

```julia
R1 = A+B
R2 = A-B
R3 = A+B-C
```

or do more complicated operations involving real and complex scalars too:

```julia
R4 = 2.0*A - B + C/(1+1im)
```

## Elementwise Operations on ITensors

ITensor objects support Julia broadcasting operations, making it quite easy to carry out element-wise operations on them in a very similar way as for regular Julia arrays. As a concrete example, consider the following ITensor initialized with random elements

```julia
i = Index(2,"i")
j = Index(3,"j")

T = randomITensor(i,j)
```

Here are some examples of basic element-wise operations we can do using Julia's dotted operator broadcasting syntax.

```julia
# Multiply every element of `T` by 2.0:
T .*= 2.0
```

```julia
# Add 1.5 to every element of T
T .+= 1.5
```

The dotted notation works for functions too:

```julia
# Replace every element in T by its absolute value:
T .= abs.(T)
```

```julia
# Replace every element in T by the number 1.0
T .= one.(T)
```

If have another ITensor `A = ITensor(j,i)`, which has the same set of indices
though possibly in a different order, then we can also do element-wise operations
involving both ITensors:

```julia
# Add elements of A and T element-wise
A .= A .+ T
```

Last but not least, it is possible to make custom functions yourself and broadcast them across elements of ITensors:

```julia
myf(x) = 1.0/(1.0+exp(-x))
T .= myf.(T)
```



## Tracing an ITensor

An important operation involving a single tensor is tracing out certain
pairs of indices. Say we have an ITensor `A` with indices `i,j,l`:

```julia
i = Index(4,"i")
j = Index(3,"j")
l = Index(4,"l")

A = randomITensor(i,j,l)
```

and we want to trace `A` by summing over the indices `i` and `l` locked together,
in other words: ``\sum_{i} A^{iji}``.

To do this in ITensor, we can use a `delta` tensor, which you can think of as
an identity operator or more generally a Kronecker delta or "hyper-edge":

![](itensor_trace_figures/delta_itensor.png)

Viewed as an array, a delta tensor has all diagonal elements equal to 1.0 and
zero otherwise.

Now we can compute the trace by contracting `A` with the delta tensor:

```julia
trA = A * delta(i,l)
```

![](itensor_trace_figures/trace_A.png)

## Write and Read an ITensor to Disk with HDF5

Saving ITensors to disk can be very useful. For example, you
might encounter a bug in your own code, and by reading the
ITensors involved from disk you can shortcut the process of
running a lengthy algorithm over many times to reproduce the bug.
Or you can save the output of an expensive calculation, such as
a DMRG calculation, and use it as a starting point for multiple
follow-up calculations such as computing time-dependent properties.

ITensors can be written to files using the HDF5 format. HDF5 offers
many benefits such as being portable across different machine types,
and offers a standard interface across various libraries and languages.

**Writing an ITensor to an HDF5 File**

Let's say you have an ITensor `T` which you have made or obtained
from a calculation. To write it to an HDF5 file named "myfile.h5"
you can use the following pattern:

```julia
using HDF5
f = h5open("myfile.h5","w")
write(f,"T",T)
close(f)
```

Above, the string "T" can actually be any string you want such as "ITensor T"
or "Result Tensor" and doesn't have to have the same name as the reference `T`.
Closing the file `f` is optional and you can also write other objects to the same
file before closing it.

**Reading an ITensor from an HDF5 File**

Say you have an HDF5 file "myfile.h5" which contains an ITensor stored as a dataset with the
name "T". (Which would be the situation if you wrote it as in the example above.)
To read this ITensor back from the HDF5 file, use the following pattern:

```julia
using HDF5
f = h5open("myfile.h5","r")
T = read(f,"T",ITensor)
close(f)
```

Note the `ITensor` argument to the read function, which tells Julia which read function
to call and how to interpret the data stored in the HDF5 dataset named "T". In the
future we might lift the requirement of providing the type and have it be detected
automatically from the data stored in the file.



