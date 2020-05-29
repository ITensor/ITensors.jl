
# Compile ITensors.jl

These are instructions for compiling ITensors.jl with 
[PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).

Run the Julia script `packagecompile.jl` with:
```
~ julia packagecompile.jl
```
This will take some time, perhaps a few minutes. (The script
will also install the PackageCompiler Julia package if you
do not already have it.)
This should create the file `sys_itensors.so` in the directory
`~/.julia/sysimages/`.
This is a version of Julia that is compiled with ITensors.
Now just start Julia with:
```
~ julia --sysimage ~/.julia/sysimages/sys_itensors.so
```
You can create an alias like:
```
~ alias julia_itensors="julia --sysimage ~/.julia/sysimages/sys_itensors.so"
```
(which you can put in your `~/.bashrc`, `~/.zshrc`, etc.).
Then if you start Julia with the command:
```
~ julia_itensors
```
you should see the startup time and JIT compilation time to
be substantially improved.
