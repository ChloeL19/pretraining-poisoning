# Public Access to Evaluation Results Viewer

## Current Public URL

**Your web app is now publicly accessible at:**

🌐 **https://subtle-classes-antarctica-radius.trycloudflare.com**

Share this URL with your team - it works from anywhere on the internet with HTTPS included!

## How It Works

- Cloudflare Tunnel creates a secure connection from your Flask app to Cloudflare's network
- No firewall changes needed
- HTTPS encryption included automatically
- Running in tmux session for persistence

## Managing the Tunnel

### Check if tunnel is running
```bash
tmux ls | grep cloudflared
```

### View tunnel status and URL
```bash
cat ~/cloudflared.log | grep trycloudflare.com
```

### Attach to tunnel session (to see live logs)
```bash
tmux attach -t cloudflared
```
Press `Ctrl+B` then `D` to detach without stopping it.

### Stop the tunnel
```bash
tmux kill-session -t cloudflared
```

### Restart the tunnel
```bash
tmux new-session -d -s cloudflared "~/.local/bin/cloudflared tunnel --url http://localhost:5000 2>&1 | tee ~/cloudflared.log"
sleep 3
cat ~/cloudflared.log | grep trycloudflare.com
```

## Important Notes

1. **URL Changes**: The free tunnel URL changes each time you restart cloudflared. For a permanent URL, see "Permanent URL Setup" below.

2. **Persistence**: The tunnel is running in a tmux session, so it will survive SSH disconnections and continue running.

3. **Server Restart**: If the server reboots, you'll need to restart the tunnel manually using the restart command above.

4. **Flask App**: Make sure the Flask app is running on localhost:5000 for the tunnel to work.

## Permanent URL Setup (Optional)

If you want a permanent URL that doesn't change, you can create a named tunnel:

```bash
# Login to Cloudflare (requires free account)
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create eval-viewer

# Get the tunnel credentials and configure
# Follow the prompts to set up a custom domain
```

This gives you a permanent subdomain like `eval-viewer.yourdomain.com`.

## Troubleshooting

### Tunnel not working?
```bash
# Check if cloudflared is running
tmux ls | grep cloudflared

# View recent logs
tail -50 ~/cloudflared.log

# Restart tunnel
tmux kill-session -t cloudflared
tmux new-session -d -s cloudflared "~/.local/bin/cloudflared tunnel --url http://localhost:5000 2>&1 | tee ~/cloudflared.log"
```

### Flask app not responding?
```bash
# Check if Flask is running
lsof -i :5000 | grep LISTEN

# Check Flask logs
tail -50 /workspace-vast/chloeloughridge/git/pretraining-poisoning/scripts/eval/view_evaluations/app.log
```

## Security

- All traffic is encrypted via HTTPS
- Cloudflare provides DDoS protection automatically
- Only HTTP traffic is proxied (no SSH or other protocols)
- Flask debug mode is acceptable behind Cloudflare Tunnel

## Cost

**Free!** Cloudflare Tunnel free tier is sufficient for small teams (2-10 people).
