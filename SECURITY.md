# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue
for a vulnerability.

(For the maintainer: this only works when private vulnerability reporting is enabled under
Settings > Code security > Private vulnerability reporting. The repository owner must switch it on.)

Include the version (`blue-onyx-prism --version`), the OS, how the service is run (service, Docker, by
hand), whether `allowed_networks` or your own TLS certificate is set, and the steps to reproduce. Expect a first answer within about a
week; this is a one-person hobby project, with no formal response time.

## Supported versions

Only the latest release gets security fixes.

## Scope notes

The security model (README, *Signing in*, *HTTPS* and *Local network only*):

- **Local network only.** Every route, on both ports, answers loopback, private and link-local
  addresses only, plus the networks listed in `allowed_networks`; others are closed at accept or
  refused with 403. Forwarding headers can only refuse.
- **Plain HTTP (port 32168) is for Blue Iris**: the detection API, `/v1/status`,
  `/v1/status/updateavailable`, a minimal `/v1/config` and static files, unauthenticated by design
  (Blue Iris cannot sign in or speak TLS). Its images and detections are unencrypted, as with
  CodeProject.AI. It accepts no credential and redirects browsers to HTTPS.
- **HTTPS (port 32169) is everything else**, behind a sign-in from every address: username and
  password accounts (all administrators; Argon2id, 15 to 128 characters, a blocklist), server-side
  sessions in `__Host-` cookies (`Secure`, `HttpOnly`, `SameSite=Strict`; 1 h idle, 12 h
  absolute), or the API token as `Authorization: Bearer` for scripts. Five wrong attempts a minute
  per address, a doubling per-account delay after 10 and disabling at 100 (device cookies keep
  their owner out of it), a bounded hashing pool. CSRF: `Origin`/`Referer` must be the same
  `https://` origin and `same-site` requests are refused; DNS rebinding: admin routes answer only
  under IP addresses, `localhost` and `allowed_hosts` names. The first account needs a one-time
  setup code that only the server's account can read.
- **The certificate is self-signed** unless you configure your own: compare its fingerprint
  (`blue-onyx-prism tls-fingerprint`) the first time. There is no HSTS, so the browser's warning can
  be clicked through.
- **Secrets** (password hashes, sessions, devices, the setup code, the API token, the TLS key) live
  in `<data root>/auth/`, readable by the service's account and administrators only; nothing logs
  or returns them.
- **Known limits:** Docker Desktop, rootless Docker and the userland proxy hide client addresses
  (bind the published ports to loopback or a LAN address); a reverse proxy's clients share its
  address for the limits; cookies are per host, not per port, so another HTTPS service on the same
  host name could read the session cookie; a Windows firewall scope of `localSubnet` is narrower
  than the app's rule.
