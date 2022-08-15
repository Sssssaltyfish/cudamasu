# cudamasu

A simple c++ header that stops intellisense from complaining about cuda stuffs.

## Usage

Just copy the header to anywhere you like, add its path to the default include path of your intellisense language server, and set this header as default include, then all is done.

Take `clangd` as an example. You should add a fragment as below to your global or project-local clangd config. Note that if latter is the case, remember to append `--enable-config` to the arguments when starts running `clangd`.

```yaml
If:
    PathMatch: [.*\.cu, .*\.cuh] # apply to all cuda source files and cuda headers
CompileFlags:
    Add: ["-isystem<where you placed the header>", "--include=cudamasu.h"]
```

In most cases `clangd` is capable of handling cuda codes, and in fact this header is modified on the very basis of clang cuda intrinsic headers.
The reason I made this hacky thing is that some day I found intellisense of cuda code was not working on another machine of mine, and that sucks.

This header is not fully tested on clangd, and not tested AT ALL on any other language servers. Any PR or issue (though I think there would be none) is welcome.
