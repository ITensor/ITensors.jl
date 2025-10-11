# vendor

Here we vendor some of our dependencies to load our own copies of them. This means:

- the functions defined in them are not shared with the "real" package they come from, so we do not see method extensions of them
- we have our own private copy of the module, allowing us to load a different version than the user

To update the dependencies, install the `vendor` environment and invoke `run.jl`:

```sh
julia --project=vendor -e 'using Pkg; Pkg.instantiate()'
julia --project=vendor vendor/run.jl
```

Note that this is based on https://github.com/JuliaTesting/ExplicitImports.jl/tree/v1.13.2/vendor.
