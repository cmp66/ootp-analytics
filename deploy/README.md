# The viewer on the Lightsail host

The viewer runs in one more container on the instance that already runs the NABL stack (Tomcat, Postgres, nginx). It is separate
from that stack: its own compose file, its own folder (`/opt/ootp-viewer`), its own nginx file. Nothing of the NABL setup changes.

- **Memory:** the container is capped at 600 MB (`mem_limit`); the viewer uses about 350 MB with one league open (a test,
  `tests/test_viewer_memory.py`, keeps every page inside that). If it ever goes over, only the viewer is restarted.
- **Data:** the league files are not in the image. Your machine pushes the processed files the viewer reads
  (`sync_files.py push --set viewer`, about 160 MB for three leagues) to the S3 bucket; the server pulls them
  (`deploy/refresh.sh`).
- **Access:** nginx asks for a user name and password (basic auth) and serves it over https.

## One-time setup on the instance

1. **Swap** (a safety net for the whole host; the instance has none):
   ```
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
2. **DNS:** a Route 53 A record `ootp.wahoosoftware.com` for the instance's static IP (the same IP as the NABL names).
3. **S3 access for the server** (see below), then `deploy/.env` on the server from `deploy/.env.example` (`chmod 600`).
4. **Code and container**, from your machine: `deploy/deploy.sh ubuntu@<ip>` (needs rsync, and Docker with the compose plugin
   on the instance, which the NABL stack already has).
5. **First data:** on the instance `/opt/ootp-viewer/deploy/refresh.sh` (after a `push --set viewer` from your machine).
6. **nginx:** `sudo cp /opt/ootp-viewer/deploy/nginx/ootp.conf /etc/nginx/conf.d/ootp.conf`, make the password file
   (`sudo apt install apache2-utils; sudo htpasswd -c /etc/nginx/ootp.htpasswd <user>`), `sudo nginx -t && sudo systemctl
   reload nginx`, then `sudo certbot --nginx -d ootp.wahoosoftware.com`.
7. **Refresh on a schedule** (optional): `*/15 * * * * /opt/ootp-viewer/deploy/refresh.sh >> /var/log/ootp-refresh.log 2>&1`.

## S3 access for the server

The bucket is only reachable from allowed IP addresses. Add the instance's static IP to that allow rule in the bucket policy.
Then make an IAM user for the server that can only read the viewer set (the access key goes into `deploy/.env`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::ootp-models",
     "Condition": {"StringLike": {"s3:prefix": ["files-viewer/*"]}}},
    {"Effect": "Allow", "Action": "s3:GetObject",
     "Resource": ["arn:aws:s3:::ootp-models/files-viewer/*", "arn:aws:s3:::ootp-models/blobs/*"]}
  ]
}
```

(`blobs/*` also holds the models; the server can read them but cannot list them or write anything.)

## Everyday

1. On your machine: `update_all.py`, then `python scripts/sync_files.py push --set viewer`.
2. The server pulls it on its next refresh (or run `deploy/refresh.sh` there). New code: `deploy/deploy.sh ubuntu@<ip>` again.
3. Look at it with `docker stats` and `free -m` once in a while; the viewer should stay near 350-450 MB.
