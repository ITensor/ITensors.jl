# Index

## Index object

```@docs
Index
```

## Index constructors

```@docs
Index(::Int)
Index(::Int, ::Union{AbstractString,TagSet})
Index()
```

## Index properties

```@docs
id(::Index)
hasid(::Index, ::ITensors.IDType)
tags(::Index)
hastags(::Index, ::Union{AbstractString,TagSet})
plev(::Index)
hasplev(::Index, ::Int)
dim(::Index)
==(::Index, ::Index)
dir(::Index)
```

## Priming and tagging methods

```@docs
prime(::Index, ::Int)
setprime(::Index, ::Int)
noprime(::Index)
settags(::Index, ::Any)
addtags(::Index, ::Any)
removetags(::Index, ::Any)
replacetags(::Index, ::Any, ::Any)
```

