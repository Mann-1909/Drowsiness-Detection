import serial

gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

while True:
    data = gps.readline().decode('ascii', errors='ignore')
    if data.startswith("$GPGGA") or data.startswith("$GPRMC"):
        print(data.strip())
