import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)

# ADD THIS LINE: Set speed to 1.35MHz (Safe for breadboards)
spi.max_speed_hz = 1350000 

def analog_read(channel):
    # Validating channel is 0-7
    if channel < 0 or channel > 7:
        return -1
    
    # SPI transaction
    r = spi.xfer2([1, (8 + channel) << 4, 0])
    adc_out = ((r[1] & 3) << 8) + r[2]
    return adc_out

print("Starting ADC read...")

while True:
    reading = analog_read(0)
    # Formula: Voltage = Reading * (Reference Voltage / Resolution)
    voltage = reading * 3.3 / 1024
    
    print("Reading=%d\tVoltage=%.2f" % (reading, voltage))
    time.sleep(1)
