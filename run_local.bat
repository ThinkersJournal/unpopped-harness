@echo off
rem Local runner: loads the MSVC environment, then forwards its arguments to cargo.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" 10.0.26100.0 >nul
cargo %*
