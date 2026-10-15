-- "Uninstall Blue Onyx Prism.app" (compiled by build-pkg.sh with osacompile, installed in
-- /Applications by the pkg). Asks whether to keep the user's data, then runs
-- scripts/uninstall_launchd.sh as root through the standard administrator password prompt.
-- The script does the work (and removes this app); keep the logic there, not here.

property appTitle : "Uninstall Blue Onyx Prism"
property uninstallScript : "/usr/local/blue-onyx-prism/scripts/uninstall_launchd.sh"
property dataDir : "/usr/local/var/blue-onyx-prism"

on run
	try
		do shell script "test -x " & quoted form of uninstallScript
	on error
		display dialog "Blue Onyx Prism is not installed on this Mac (" & uninstallScript & " is missing)." buttons {"OK"} default button "OK" with title appTitle with icon caution
		return
	end try

	set answer to button returned of (display dialog "Uninstall Blue Onyx Prism?" & return & return & "This stops the service and removes the program." & return & return & "Your configuration, models and logs (" & dataDir & ") can be kept for a later reinstall, or deleted too." buttons {"Cancel", "Delete Everything", "Keep My Data"} default button "Keep My Data" cancel button "Cancel" with title appTitle with icon caution)

	set flags to ""
	if answer is "Delete Everything" then set flags to " --purge"

	try
		do shell script quoted form of uninstallScript & flags with prompt "Blue Onyx Prism needs an administrator password to remove its background service." with administrator privileges
	on error errText number errNum
		if errNum is -128 then return -- password prompt cancelled
		display dialog "Uninstall failed:" & return & return & errText buttons {"OK"} default button "OK" with title appTitle with icon stop
		return
	end try

	if answer is "Delete Everything" then
		set doneText to "Blue Onyx Prism and all its data were removed."
	else
		set doneText to "Blue Onyx Prism was removed. Your configuration, models and logs were kept in " & dataDir & "."
	end if
	display dialog doneText buttons {"OK"} default button "OK" with title appTitle with icon note
end run
