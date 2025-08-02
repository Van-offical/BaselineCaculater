@echo off
setlocal enabledelayedexpansion

if not exist "requirements.txt" (
    echo Error: requirements.txt file not found in the current directory
    pause
    exit /b 1
)

for /f "tokens=*" %%i in (requirements.txt) do (
    pip show "%%i" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Installing library: %%i
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "%%i"
        if !errorlevel! neq 0 (
            echo Installation failed: %%i
            pause
            exit /b 1
        )
    ) else (
        echo Library already exists: %%i
    )
)

echo Dependency check and installation completed!
pause