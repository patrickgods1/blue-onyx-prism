//! The one bounded runner for short helper programs (sysctl, netstat, lsof, launchctl,
//! PowerShell, `id`; audit 2026-11): stdout and stderr are drained on threads while waiting so a
//! full pipe never blocks the child, output is capped, the child's process tree is killed at the
//! deadline, and a grandchild that keeps a pipe open cannot hang the caller.
