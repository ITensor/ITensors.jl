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

