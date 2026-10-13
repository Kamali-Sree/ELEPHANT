@echo off
title Push Project to GitHub
echo ========================================================
echo   Pushing ELEPHANT Project to GitHub
echo   Branch: feature/elephant-system
echo   Remote: https://github.com/Kamali-Sree/ELEPHANT.git
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/4] Ensuring branch 'feature/elephant-system'...
git checkout -B feature/elephant-system
echo.

echo [2/4] Staging all added modules, models, frontend, backend, and data assets...
git add -A
git status --short
echo.

echo [3/4] Creating commit...
git commit -m "docs: comprehensive system documentation, architecture, and module guide"
echo.

echo [4/4] Pushing branch 'feature/elephant-system' to GitHub...
echo (If prompted, complete GitHub collaborator sign-in in browser or credential prompt)
git push -u origin feature/elephant-system

echo.
if %ERRORLEVEL% equ 0 (
    echo ========================================================
    echo  SUCCESS: All modules and files pushed successfully!
    echo  Branch:  feature/elephant-system
    echo  URL:     https://github.com/Kamali-Sree/ELEPHANT/tree/feature/elephant-system
    echo ========================================================
) else (
    echo ========================================================
    echo  Push encountered an issue. See git output above.
    echo ========================================================
)
echo.
pause
