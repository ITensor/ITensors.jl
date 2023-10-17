# ITensor Development Roadmap

As of October 9, 2023: Things may change over time, please reach out if this is outdated.

## State of the ecosystem

- NDTensors.jl
- ITensors.jl
- ITensorNetworks.jl
- ITensorTDVP.jl
https://github.com/ITensor/ITensorTDVP.jl (has an MPS linear solver)

## NDTensors.jl plan

## ITensors.jl plan

## ITensorNetworks.jl plan

## ITensorQTT.jl plan
https://github.com/mtfishman/ITensorQTT.jl (has functionality for making some differential operators as MPOs, as well as other QTT operators like QFT, interpolations, etc.)

## ITensorNumericalAnalysis.jl

We already have some of those features in:

https://github.com/mtfishman/ITensorNetworks.jl (has a TTN linear solver)
https://github.com/mtfishman/ITensorQTT.jl (has functionality for making some differential operators as MPOs, as well as other QTT operators like QFT, interpolations, etc.)

and Miles is working on a modern TT-cross implementation (he has a private one already, but we are discussing integrating it into the modern tensor network solver interface being developed in ITensorNetworks.jl). Additionally, we are planning on using ITensorNetworks.jl as the basis for all major future tensor network algorithm developments (including next-gen linear solvers and QTT work), since it has support for tree tensor networks and more general tensor networks. So we have been discussing internally plans to make a QTT library based on ITensorNetworks.jl and there is a prototype for that in progress by some postdocs at the Flatiron (which we think we will call ITensorNumericalAnalysis.jl), but we are in a discussion phase about how to design a nice and general quantized tensor network type that encodes the dimensions and bits in the type in a flexible way (so for example you can use a variety of different tensor network backends, bit orderings, and arbitrary dimensions).

An issue I have had recently is that I have been completely swamped reviewing code and working on collaborations and have not had time to do my own coding. However, in general it is best if different groups tried to align their efforts and didn't repeat work unnecessarily, so I appreciate that you reached out.

A strategy we have taken recently is developing separate packages for certain smaller features which then can get contributed into either a core package (like ITensors.jl or ITensorNetworks.jl) or combined into a larger external package. I think a challenge here is that there are many moving parts and pieces in various stages of development, and it can be challenging to coordinate developing "core" library features across different groups without me becoming a bottleneck in the development (and overwhelming me with work). I am the primary one who reviews code that gets contributed to the ITensor library and we generally have very particular standards and requirements for what kind of code we want in core libraries, as well as larger visions or design plans that take longer to come into fruition than a standard research code. There is a big difference between having something that "just works" for a research project and something that will last 5 or 10 years, has a nice generic and modular design, is well integrated into the core library in a way that avoids code duplication, and can be built off of for future research and software development even after the initial author of the code has moved on to other things. That kind of development can take a lot longer, and also is a lot of work for us as the core developers.
