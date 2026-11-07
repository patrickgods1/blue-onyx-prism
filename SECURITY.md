# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue
for a vulnerability.

(For the maintainer: this only works when private vulnerability reporting is enabled under
Settings > Code security > Private vulnerability reporting. The repository owner must switch it on.)

Include the version (`blue-onyx-prism --version`), the OS, how the service is run (service, Docker, by
hand), whether `allow_remote_admin` is on, and the steps to reproduce. Expect a first answer within about a
week; this is a one-person hobby project, with no formal response time.

## Supported versions

Only the latest release gets security fixes.

## Scope notes

- The service is meant for a trusted LAN. Detection routes (`/v1/vision/*`, `/v1/status`, `/prometheus`)
  are open by design, as Blue Iris needs them; admin routes answer this machine, or remote peers that show the
  admin token while `allow_remote_admin` is on. The token and cookie travel over plain HTTP, so put TLS in front
  of remote admin or keep it on a trusted network (see the README, *Remote access*).
- A local reverse proxy that adds no forwarding header looks like this machine; the README gives the
  header line to add.
