name: TestITensorUnicodePlots
on:
  push:
    branches:
      - main
    tags: '*'
  pull_request:
jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }} - ${{ matrix.threads }} thread(s)
    runs-on: ${{ matrix.os }}
    env:
      JULIA_NUM_THREADS: ${{ matrix.threads }}
    strategy:
      matrix:
        version:
          - '1.3'
          - '1'
        os:
          - ubuntu-latest
        threads:
          - '1'
        arch:
          - x64
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@latest
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      #- uses: julia-actions/julia-buildpkg@latest
      - name: Install Julia dependencies
        shell: julia --project=monorepo {0}
        run: |
          using Pkg;
          # dev mono repo versions
          pkg"dev . ./ITensorUnicodePlots"
      #- uses: julia-actions/julia-runtest@latest
      - name: Run the tests
        continue-on-error: true
        run: >
          julia --project=monorepo -e 'using Pkg; Pkg.test("ITensorUnicodePlots", coverage=true)'
          && echo "TESTS_SUCCESSFUL=true" >> $GITHUB_ENV 
      - uses: julia-actions/julia-uploadcodecov@latest
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
