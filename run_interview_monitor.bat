@echo off
chcp 65001 >nul
title Interview Monitor System

echo.
echo ================================================================
echo 🎯 INTERVIEW MONITOR SYSTEM
echo ================================================================
echo AI-powered interview candidate analysis with computer vision
echo ================================================================
echo.

:menu
echo What would you like to do?
echo.
echo 1. Run Demo Mode (no camera/microphone required)
echo 2. Run Full Interview Monitor (requires camera/microphone)
echo 3. Run HUD Interface Only
echo 4. Run System Tests
echo 5. Show Help
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto monitor
if "%choice%"=="3" goto hud
if "%choice%"=="4" goto test
if "%choice%"=="5" goto help
if "%choice%"=="6" goto exit
echo Invalid choice. Please enter 1-6.
goto menu

:demo
echo.
echo 🎭 Starting Demo Mode...
python main.py --demo
goto end

:monitor
echo.
echo 🎥 Starting Full Interview Monitor...
python main.py --monitor
goto end

:hud
echo.
echo 🖥️ Starting HUD Interface...
python main.py --hud
goto end

:test
echo.
echo 🧪 Running System Tests...
python main.py --test
goto end

:help
echo.
python main.py --help
echo.
pause
goto menu

:exit
echo.
echo 👋 Goodbye!
echo.
pause
exit

:end
echo.
echo Press any key to return to menu...
pause >nul
goto menu
