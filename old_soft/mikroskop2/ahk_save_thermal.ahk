SetTitleMatchMode, 2
SetTitleMatchMode, Fast

winTitle := "Optris PIX Connect (Rel. 3.19.3107.0)"

IfWinExist, %winTitle%
{
	WinActivate
	Sleep, 500
	Send {F1}
	Sleep, 1000
	Send, %1%
	Sleep, 300
	Send, .csv
	Sleep, 500
	Send, {Enter}
}