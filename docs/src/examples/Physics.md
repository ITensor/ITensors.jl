# Physics (SiteType) System Examples

## Obtaining a Predefined Operator

Given an Index carrying a "physical" tag such as "Qubit", "S=1/2", "Boson", etc.
there are a set of pre-defined operators for each tag. The entire set of operators
can be found in the section [SiteTypes Included with ITensor](@ref).

If you have an Index `s` carrying a "S=1/2" tag, for example, you can obtain the "Sz"
operator like this:
```julia
op("Sz",s)
```

Usually indices with physical tags come from an array of indices returned from the `siteinds` function
```julia
sites = siteinds("S=1/2",N)
```
in which case one might want the "Sz" operator on site 4
```julia
Sz4 = op("Sz",sites[4])
```

## Make a Custom Operator from a Matrix

The `op` function can be passed any matrix, as long as it has the correct dimensions,
and it will make this into an ITensor representing the operator with the corresponding
matrix elements.

For example, if we have a two-dimensional Index `s` we could make the "Sz" operator ourselves from
the matrix
```julia
M = [1/2 0 ; 0 -1/2]
```
by calling
```julia
Sz = op(M,s)
```


## [Making a Custom op Definition](@id custom_op)

The function `op` is used to obtain operators defined for a 
given "site type". ITensor includes pre-defined site types such
as "S=1/2", "S=1", "Electron" and others. Or you can define your own site type
as discussed in detail in the code examples further below.

**Extending op Function Definitions**

Perhaps the most common part of the site type system one wishes to extend
are the various `op` or `op!` function overloads which allow code like

```julia
s = siteind("S=1/2")
Sz = op("Sz",s)
```

to automatically create the ``S^z`` operator for an Index `s` based on the 
`"S=1/2"` tag it carries. A major reason to define such `op` overloads
is to allow the OpSum system to recognize new operator names, as
discussed more below.

Let's see how to introduce a new operator name into the ITensor site type
system for this existing site type of `"S=1/2"`. The operator we will
introduce is the projector onto the up spin state ``P_\uparrow`` which
we will denote with the string `"Pup"`. 

As a matrix acting on the space ``\{ |\!\uparrow\rangle, |\!\downarrow\rangle \}``,
the ``P_\uparrow`` operator is given by

```math
\begin{aligned}

P_\uparrow &= 
\begin{bmatrix}
 1 &  0 \\
 0  & 0 \\
\end{bmatrix}

\end{aligned}
```

To add this operator to the ITensor `op` system, we just need to introduce the following
code

```julia
using ITensors

ITensors.op(::OpName"Pup",::SiteType"S=1/2") =
 [1 0
  0 0]
```

This code can be defined anywhere, such as in your own personal application code and does 
not have to be put into the ITensor library source code.

Note that we have to name the function `ITensors.op` and not just `op` so that it overloads
other functions of the name `op` inside the ITensors module. 

Having defined the above code, we can now do things like

```julia
s = siteind("S=1/2")
Pup = op("Pup",s)
```

to obtain the `"Pup"` operator for our `"S=1/2"` Index `s`. Or we can do a similar
thing for an array of site indices:

```julia
N = 40
s = siteinds("S=1/2",N)
Pup1 = op("Pup",s[1])
Pup3 = op("Pup",s[3])
```

Note that for the `"Qudit"`/`"Boson"` site types, you have to define your overload
of `op` with the dimension of the local Hilbert space, for example:
```julia
using ITensors

function ITensors.op(::OpName"P1", ::SiteType"Boson", d::Int)
  o = zeros(d, d)
  o[1, 1] = 1
  return o
end
```
Alternatively you could use Julia's [array comprehension](https://docs.julialang.org/en/v1/manual/arrays/#man-comprehensions) syntax:
```julia
ITensors.op(::OpName"P1", ::SiteType"Boson", d::Int) =
  [(i == j == 1) ? 1.0 : 0.0 for i in 1:d, j in 1:d]
```

**Using Custom Operators in OpSum**

A key use of these `op` system extensions is allowing additional operator names to
be recognized by the OpSum system for constructing matrix product operator (MPO)
tensor networks. With the code above defining the `"Pup"` operator, we are now 
allowed to use this operator name in any OpSum code involving `"S=1/2"` site 
indices.

For example, we could now make an OpSum involving our custom operator such as:

```julia
N = 100
sites = siteinds("S=1/2",N)
os = OpSum()
for n=1:N
  os += "Pup",n
end
P = MPO(os,sites)
```

This code makes an MPO `P` which is just the sum of a spin-up projection operator
acting on every site.


## Making a Custom state Definition

The function `state` is used to define states (single-site wavefunctions)
that sites can be in. For example, the "Qubit" site type includes 
definitions for the "0" and "1" states as well as the "+" (eigenstate of X operator)
state. The "S=1/2" site type includes definitions for the "Up" and "Dn" (down) states.

Say we want to define a new state for the "Electron" site type called "+", which has
the meaning of one electron with its spin in the +X direction. First let's review
the existing state definitions:
```julia
ITensors.state(::StateName"Emp", ::SiteType"Electron") = [1.0, 0, 0, 0]
ITensors.state(::StateName"Up", ::SiteType"Electron") = [0.0, 1, 0, 0]
ITensors.state(::StateName"Dn", ::SiteType"Electron") = [0.0, 0, 1, 0]
ITensors.state(::StateName"UpDn", ::SiteType"Electron") = [0.0, 0, 0, 1]
```
As we can see, the four settings of an "Electron" index correspond to the states
``|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle``.

So we can define our new state "+" as follows:
```julia
ITensors.state(::StateName"+", ::SiteType"Electron") = [0, 1/sqrt(2), 1/sqrt(2), 0]
```
which makes the state
```math
|+\rangle = \frac{1}{\sqrt{2}} |\uparrow\rangle + \frac{1}{\sqrt{2}} |\downarrow\rangle
```

Having defined this overload of `state`, if we have an Index of type "Electron"
we can obtain our new state for it by doing
```julia
s = siteind("Electron")
plus = state("+",s)
```
We can also use this new state definition in other ITensor features such as 
the MPS constructor taking an array of state names.


## Make a Custom Local Hilbert Space / Physical Degree of Freedom

ITensor provides support for a range of common local Hilbert space types, 
or physical degrees of freedom, such as S=1/2 and S=1 spins; spinless and spinful
fermions; and more.

However, there can be many cases where you need to make custom
degrees of freedom. You might be working with an
exotic system, such as ``Z_N`` parafermions for example, or need
to customize other defaults provided by ITensor.

In ITensor, such a customization is done by overloading functions
on specially designated Index tags. 
Below we give an brief introduction by example of how to make
such custom Index site types in ITensor. 
Other code formulas following this one explain how to build on this
example to expand the capabilities of your custom site type such as
adding support for quantum number (QN) conservation and defining
custom mappings of strings to states.

Throughout we will focus on the example of ``S=3/2`` spins. These
are spins taking the ``S^z`` values of ``+3/2,+1/2,-1/2,-3/2``.
So as tensor indices, they are indices of dimension 4.

The key operators we will make for this example are ``S^z``, ``S^+``,
and ``S^-``, which are defined as:

```math
\begin{aligned}
S^z &= 
\begin{bmatrix}
3/2 &  0  &  0  &  0 \\
 0  & 1/2 &  0  &  0 \\
 0  &  0  &-1/2 &  0 \\
 0  &  0  &  0  &-3/2\\
\end{bmatrix} \\

S^+ & = 
\begin{bmatrix}
 0  &  \sqrt{3}  &  0  &  0 \\
 0  &  0  &  2  &  0 \\
 0  &  0  &  0  &  \sqrt{3} \\
 0  &  0  &  0  &  0 \\
\end{bmatrix} \\

S^- & = 
\begin{bmatrix}
 0  &  0 &  0  &  0 \\
 \sqrt{3}  &  0  &  0  &  0 \\
 0  &  2  &  0  &  0  \\
 0  &  0  &  \sqrt{3}  &  0 \\
\end{bmatrix} \\
\end{aligned}
```

**Code Preview**

First let's see the minimal code needed to define and use this new
``S=3/2`` site type, then we will discuss what each part of
the code is doing.

```julia
using ITensors

ITensors.space(::SiteType"S=3/2") = 4

ITensors.op(::OpName"Sz",::SiteType"S=3/2") =
  [+3/2   0    0    0
     0  +1/2   0    0 
     0    0  -1/2   0
     0    0    0  -3/2]

ITensors.op(::OpName"S+",::SiteType"S=3/2") =
  [0  √3  0  0
   0   0  2  0
   0   0  0 √3
   0   0  0  0] 

ITensors.op(::OpName"S-",::SiteType"S=3/2") =
  [0   0  0   0
   √3  0  0   0
   0   2  0   0
   0   0  √3  0] 

```

Now let's look at each part of the code above.

**The SiteType**

The most important aspect of this code is a special type, known as a `SiteType`,
which is a type made from a string. The string of interest here will be an Index
tag. In the code above, the `SiteType` we are using is

```julia
SiteType"S=3/2"
```

What is the purpose of a `SiteType`? The answer is that we would like to be 
able to select different functions to call on an ITensor Index based on what tags
it has, but that is not directly possible in Julia or indeed most languages. 
However, if we can map a tag
to a type in the Julia type system, we can create function overloads for that type.
ITensor does this for certain functions for you, and we will discuss a few of these
functions below. So if the code encounters an Index such as `Index(4,"S=3/2")` it can 
call these functions which are specialized for indices carrying the `"S=3/2"` tag. 

**The space Function**

One of the overloadable `SiteType` functions is `space`, whose job is to 
describe the vector space corresponding to that site type. For our
`SiteType"S=3/2"` overload of `space`, which gets called for any Index 
carrying the `"S=3/2"` tag, the definition is

```julia
ITensors.space(::SiteType"S=3/2") = 4
```

Note that the function name is prepended with `ITensors.` before `space`.
This prefix makes sure the function is overloading other versions of the `space`
inside the `ITensors` module.

The only information needed about the vector space of a `"S=3/2"` Index in
this example is that it is of dimension four. So the `space` function returns
the integer `4`. We will see in more advanced examples that the returned value
can instead be an array which specifies not only the dimension of a `"S=3/2"`
Index, but also additional subspace structure it has corresponding to quantum
numbers.

After defining this `space` function, you can just write code like:

```julia
s = siteind("S=3/2")
```

to obtain a single `"S=3/2"` Index, or write code like

```julia
N = 100
sites = siteinds("S=3/2",N)
```

to obtain an array of N `"S=3/2"` indices. The custom `space` function
will be used to determine the dimension of these indices, and the `siteind`
or `siteinds` functions provided by ITensor will help with extra things like
putting other Index tags that are conventional for site indices.

**The op Function**

The `op` function lets you define custom local operators associated
to the physical degrees of freedom of your `SiteType`. Then for example 
you can use indices carrying your custom tag with OpSum and the 
OpSum system will know how to automatically convert names of operators
such as `"Sz"` or `"S+"` into ITensors so that it can make an actual MPO.

In our example above, we defined this function for the case of the `"Sz"`
operator as:

```@example S32
using ITensors # hide

ITensors.op(::OpName"Sz",::SiteType"S=3/2") =
  [+3/2   0    0    0
     0  +1/2   0    0 
     0    0  -1/2   0
     0    0    0  -3/2]
```

As you can see, the function is passed two objects: an `OpName` and a `SiteType`.
The strings `"Sz"` and `"S=3/2"` are also part of the type of these objects, and 
have the meaning of which operator name we are defining and which site type these
operators are defined for.

The body of this overload of `ITensors.op` constructs and returns a Julia matrix
which gives the matrix elements of the operator we are defining.

Once this function is defined, and if you have an Index such as

```@example S32; continued = true
s = Index(4,"S=3/2")
```

then, for example, you can get the `"Sz"` operator for this Index 
and print it out by doing:

```@example S32
Sz = op("Sz",s)
println(Sz)
```

Again, through the magic of the `SiteType`
system, the ITensor library takes your Index, reads off its tags, 
notices that one of them is `"S=3/2"`, and converts this into the type 
`SiteType"S=3/2"` in order to call the specialized function `ITensors.op` defined above.

You can use the `op` function yourself with a set of site indices created from
the `siteinds` function like this:

```julia
N = 100
sites = siteinds("S=3/2",N)
Sz1 = op("Sz",sites[1])
Sp3 = op("S+",sites[3])
```

Alternatively, you can write the lines of code above in the style
of `Sz1 = op("Sz",sites,1)`.

This same `op` function is used inside of OpSum (formerly called AutoMPO) 
when it converts its input into
an actual MPO. So by defining custom operator names you can pass any of these
operator names into OpSum and it will know how to use these operators.

**Further Steps**

See how the built-in site types are defined inside the ITensor library:
* [S=1/2 sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/spinhalf.jl) - Dimension 2 local Hilbert space. Similar to the `"Qubit"` site type, shares many of the same operator definitions.
* [Qubit sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/qubit.jl) - Dimension 2 local Hilbert space. Similar to the `"S=1/2"` site type, shares many of the same operator definitions.
* [S=1 sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/spinone.jl) - Dimension 3 local Hilbert space.
* [Fermion sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/fermion.jl) - Dimension 2 local Hilbert space. Spinless fermion site type.
* [Electron sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/electron.jl) - Dimension 4 local Hilbert space. Spinfull fermion site type.
* [tJ sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/tj.jl) - Dimension 3 local Hilbert space. Spinfull fermion site type but without a doubly occupied state in the Hilbert space.
* [Boson sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/boson.jl) - General d-dimensional local Hilbert space. Shares the same operator definitions as the `"Qudit"` site type.
* [Qudit sites](https://github.com/ITensor/ITensors.jl/blob/main/src/physics/site_types/qudit.jl) - General d-dimensional local Hilbert space. Generalization of the `"Qubit"` site type, shares the same operator definitions as the ``Boson`` site type.


## Make a Custom Local Hilbert Space with QNs

In the previous example above, we discussed the basic,
minimal code needed to define a custom local Hilbert space, using the example
of a ``S=3/2`` spin Hilbert space. In those examples, the `space` function
defining the vector space of a ``S=3/2`` spin only provides the dimension of 
the space. But the Hilbert space of a ``S=3/2`` spin has additional structure, which
is that each of its four subspaces (each of dimension 1) can be labeled by 
a different ``S^z`` quantum number.

In this code formula we will include this extra quantum information in the 
definition of the space of a ``S=3/2`` spin.

**Code Preview**

First let's see the minimal code needed to add the option for including
quantum numbers of our ``S=3/2`` site type, then we will discuss what each part of
the code is doing.

```julia
using ITensors

function ITensors.space(::SiteType"S=3/2";
                        conserve_qns=false)
  if conserve_qns
    return [QN("Sz",3)=>1,QN("Sz",1)=>1,
            QN("Sz",-1)=>1,QN("Sz",-3)=>1]
  end
  return 4
end

ITensors.op(::OpName"Sz",::SiteType"S=3/2") =
  [+3/2   0    0    0
     0  +1/2   0    0 
     0    0  -1/2   0
     0    0    0  -3/2]

ITensors.op(::OpName"S+",::SiteType"S=3/2") =
  [0  √3  0  0
   0   0  2  0
   0   0  0 √3
   0   0  0  0] 

ITensors.op(::OpName"S-",::SiteType"S=3/2") =
  [0   0  0   0
   √3  0  0   0
   0   2  0   0
   0   0  √3  0] 


```

Now let's look at each part of the code above.

**The space function**

In the previous code example above, we discussed 
that the function `space` tells the ITensor library the basic information about how
to construct an Index associated with a special Index tag, in this case the tag `"S=3/2"`.
As in that code formula, if the user does not request that quantum numbers be included
(the case `conserve_qns=false`) then all that the `space` function returns is the number
4, indicating that a `"S=3/2"` Index should be of dimension 4.

But if the `conserve_qns` keyword argument gets set to `true`, the `space` function we
defined above returns an array of `QN=>Int` pairs. (The notation `a=>b` in Julia constructs
a `Pair` object.) Each pair in the array denotes a subspace.
The `QN` part of each pair says what quantum number the subspace has, and the integer following
it indicates the dimension of the subspace.

After defining the `space` function this way, you can write code like:

```julia
s = siteind("S=3/2"; conserve_qns=true)
```

to obtain a single `"S=3/2"` Index which carries quantum number information.
The `siteind` function built into ITensor relies on your custom `space` function
to ask how to construct a `"S=3/2"` Index but also includes some other Index tags
which are conventional for all site indices.

You can now also call code like:

```julia
N = 100
sites = siteinds("S=3/2",N; conserve_qns=true)
```

to obtain an array of N `"S=3/2"` indices which carry quantum numbers.

**The op Function in the Quantum Number Case**

Note that the `op` function overloads are exactly the same as for the
more basic case of defining an `"S=3/2"` Index type that does not carry
quantum numbers. There is no need to upgrade any of the `op` functions 
for the QN-conserving case. 
The reason is that all QN, block-sparse information
about an ITensor is deduced from the indices of the tensor, and setting elements
of such tensors does not require any other special code. 

However, only operators which have a well-defined QN flux---meaning they always
change the quantum number of a state they act on by a well-defined amount---can
be used in practice in the case of QN conservation. Attempting to build an operator, or any ITensor,
without a well-defined QN flux out of QN-conserving indices will result in a run time error.
An example of an operator that would lead to such an error would be the "Sx" spin operator
since it alternately increases ``S^z`` or decreases ``S^z`` depending on the state it acts
on, thus it does not have a well-defined QN flux. But it is perfectly fine to define an
`op` overload for the "Sx" operator and to make this operator when working with dense, 
non-QN-conserving ITensors or when ``S^z`` is not conserved.


