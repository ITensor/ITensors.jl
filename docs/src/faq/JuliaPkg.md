# Julia Package Manager Frequently Asked Questions

## What if I can't upgrade ITensors.jl to the latest version?

Sometimes you may find that doing `] update ITensors` or equivalently doing `] up ITensors` within
Julia package manager mode doesn't result in the ITensors package
actually being upgraded. You may see that the current version
you have remains stuck to a version that is lower than the latest one which you 
can [check here](https://github.com/ITensor/ITensors.jl).

What is most likely going on is that you have other packages installed which
are blocking ITensors from being updated.

To get more information into which packages may be doing this, and what versions
they are requiring, you can do the following. First [look up the latest version of ITensors.jl](https://github.com/ITensor/ITensors.jl). Let's say for this example that it is `v0.3.0`. 

Next, input the following command while in package manager mode:

```
julia> ]
pkg> add ITensors@v0.3.0
```

If the package manager cannot update to this version, it will list all of the other packages that are blocking this from happening and give information about why. To go into a little more depth, each package has a compatibility or "compat" entry in its Project.toml file which says which versions of the ITensors package it is compatible with. If these versions do not include the latest one, perhaps because the package has not been updated, then it can block the ITensors package from being updated on your system.

Generally the solution is to just update each of these packages, then try again to update ITensors. If that does not work, then check the following
* Are any of the blocking packages in "dev mode" meaning you called `dev PackageName` on them in the past? Try doing `free PackageName` if so to bring them out of dev mode.
* Are any of the blocking packages unregistered packages that were installed through a GitHub repo link? If so, you may need to do something like `add https://github.com/Org/PackageName#main` to force update that package to the latest code available on its main branch.

If you still can't get the ITensors package update, feel free to [post a question](https://itensor.org/support) or [contact us](https://itensor.org/about.html#collaboration) for help.

