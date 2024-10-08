@ECHO off

cd C:\Users\willf\OneDrive\Telecom\2023-2024\SD201\Lab\Project\lab_data\lab01
CALL conda activate ds

CALL :testCode 1
CALL :testCode 2
CALL :testCode 3
CALL :testCode 4
CALL :testCode 5
CALL :testCode 6
CALL :testCode 7
CALL :testCode 8
CALL :testCode 9
CALL :testCode 10
CALL :testCode 11

ECHO.

CALL conda deactivate

EXIT /B %ERRORLEVEL%

:printDiv
set line=------------------------------------------------------
ECHO. & ECHO %line% & ECHO.
EXIT /B 0

:testCode
CALL :printDiv
ECHO %~1
CALL :printDiv
python .\main.py debug %~1
python .\check_results.py debug %~1
ECHO.
python .\main.py eval %~1
python .\check_results.py eval %~1
EXIT /B 0
