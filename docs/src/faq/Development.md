# ITensor Development Frequently Asked Questions

## What are the steps to contribute code to ITensor?

1. Please contact us (support at itensor.org) if you are 
   planning to submit a major
   contribution (more than a few lines of code, say).
   If so, we would like to discuss your plan and design
   before you spend significant time on it, to increase
   the chances we will merge your pull request.

2. Fork the [ITensors.jl](https://github.com/ITensor/ITensors.jl) Github repo,
   create a new branch and make changes (commits) on that branch.

3. Run the ITensor unit tests by going into the test/ folder and running
   `julia runtests.jl`. To run individual test scripts, start a Julia 
   REPL (interactive terminal) session and include each script, such as
   `include("itensor.jl")`.

3. Push your new branch and changes to your forked repo. 
   Github will give you the option to make a 
   pull request (PR) out of your branch that will be submitted to us, and which
   you can view under the list of ITensors.jl pull requests.
   If your PR's tests pass and we approve your changes, we will merge it or 
   ask you to merge it. If you merge your PR, _please use the Squash and Merge_ option. 
   We may also ask you to make more changes to bring your PR in line with our
   design goals or technical requirements.


