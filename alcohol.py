import spidev
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

buzzer_pin = 17
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

# ---------------- HEART RATE VARIABLES ----------------
threshold = 517        # Adjust based on your signal
min_interval = 0.35    # Minimum 350ms between beats (avoid double detection)
last_beat = 0
bpm_values = []

# -------------------------------------------------------
def analog_read(channel):
    if channel < 0 or channel > 7:
        return -1

    r = spi.xfer2([1, (8 + channel) << 4, 0])
    adc_value = ((r[1] & 3) << 8) + r[2]
    return adc_value

print("Starting ADC + BPM Monitor...")
i = 0

while True:
    reading_alcohol = analog_read(0)
    reading_heart = analog_read(1)
    reading_temp = analog_read(2)

    voltage1 = reading_alcohol * 3.3 / 1024
    voltage2 = reading_heart * 3.3 / 1024
    voltage3 = reading_temp * 3.3 / 1024

    # print(i)
    # print("Alcohol Reading=%d\tVoltage=%.2f" % (reading_alcohol, voltage1))
    # print("Heart Raw Reading=%d\tVoltage=%.2f" % (reading_heart, voltage2))

    # ----------------- BPM CALCULATION -----------------
    current_time = time.time()

    if reading_heart > threshold:  
        if (current_time - last_beat) > min_interval:
            ibi = current_time - last_beat
            last_beat = current_time

            bpm = 60 / ibi
            bpm_values.append(bpm)

            if len(bpm_values) > 5:
                bpm_values.pop(0)

            avg_bpm = int(sum(bpm_values) / len(bpm_values))
            print("♥ Beat detected!")
            print("IBI =", ibi)
            print("BPM =", avg_bpm)

            # Example alert
            if avg_bpm > 120 or avg_bpm < 45:
                GPIO.output(buzzer_pin, GPIO.HIGH)
            else:
                GPIO.output(buzzer_pin, GPIO.LOW)

    # print("=======================================================")

    time.sleep(0.02)  # Faster sampling gives better BPM accuracy
    i += 1
