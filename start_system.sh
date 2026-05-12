#!/bin/bash

# 1. Wait for the internet to stabilize (hotspots can be slow to connect)
sleep 20

# 2. Add your authtoken (Ensure you replace the text below with your real token)
# Go to https://dashboard.ngrok.com/get-started/your-authtoken to find it
/home/pi/Desktop/DDD/ngrok config add-authtoken 3C9pXA1eT7QTbLWBUhA5isI4Slw_46FkAtKoLmjVXdkaLkVme

# 3. Clear any old stuck Ngrok processes
sudo pkill ngrok

# 4. Start the Ngrok tunnel and send logs to the system logger for debugging
(/home/pi/Desktop/DDD/ngrok http --domain=legislate-obituary-jolliness.ngrok-free.dev 5000 2>&1 | logger -t ngrok) &

# 5. Move to the project folder and start the main safety system
cd /home/pi/Desktop/DDD
python3 main.py