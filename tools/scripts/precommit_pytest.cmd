@echo off
REM Run pytest with the MSVC environment the recipes use, for the pre-commit hook.
REM
REM Why this exists rather than the environment being set in the hook: one test the tree carries
REM builds a torch C++/CUDA extension at run time (tests/convert/quantization/test_stored_scale.py,
REM through tools/convert/quantization/stored_scale.py), and torch locates the compiler with
REM `where cl`. cl.exe is not on PATH in a plain shell on any Windows machine, so that test failed
REM with "Could not find files for the given pattern(s)" -- an environment error that reads as a test
REM failure. Without a GPU the same test skips (torch.cuda.is_available() is False), which is why CI
REM never needed this.
REM
REM The compiler environment comes from vcvars64.bat, located through vswhere so this does not pin a
REM Visual Studio edition or version, with the edition this machine has as a fallback.
setlocal
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo pre-commit: pytest  ^(no vcvars64.bat found; a test that builds an extension may fail^) 1>&2
) else (
  call "%VCVARS%" >nul 2>&1
)
set "PY=%NINFER_PYTHON%"
if not defined PY set "PY=C:/vllm-env/Scripts/python.exe"
"%PY%" -m pytest %*
exit /b %ERRORLEVEL%
