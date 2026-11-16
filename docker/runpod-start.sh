#!/usr/bin/env bash
set -euo pipefail

# RunPod injects PUBLIC_KEY for SSH access. Prepare it before starting sshd.
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A

# Keep sshd in the foreground so the container remains alive and RunPod
# can establish terminal access.
exec /usr/sbin/sshd -D -e
