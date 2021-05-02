# Sweeps

```@docs
Sweeps
Sweeps(nsw::Int, d::AbstractMatrix)
```

## Modifying Sweeps Objects

```@docs
setmaxdim!
setcutoff!
setnoise!
setmindim!
```

## Getting Sweeps Object Data

```@docs
nsweep(sw::Sweeps)
maxdim(sw::Sweeps,n::Int)
cutoff(sw::Sweeps,n::Int)
noise(sw::Sweeps,n::Int)
mindim(sw::Sweeps,n::Int)
```

