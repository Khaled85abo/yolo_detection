import os
# Set correct platform and GPIO base address for Raspberry Pi 5
os.environ.pop('BLINKA_FT232H', None)
os.environ['BLINKA_FORCEBOARD'] = 'GENERIC_LINUX_PC'
os.environ['BLINKA_FORCECHIP'] = 'BCM2XXX'
os.environ['BLINKA_PLATFORM_OVERRIDE'] = 'linux'
os.environ['PIGPIO_ADDR'] = 'localhost'
os.environ['GPIOZERO_PIN_FACTORY'] = 'rpigpio'
os.environ['PI_PERIPHERAL_BASE'] = '0x7c000000'  # Specific for Raspberry Pi 5

# Import required libraries
import RPi.GPIO as GPIO
import digitalio
import board
import adafruit_ili9341
import time
from PIL import Image, ImageDraw, ImageFont
import busio
from microcontroller.pin import Pin

try:
    # Initialize GPIO
    print("Initializing GPIO...")
    GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
    GPIO.setwarnings(False)
    
    # Setup GPIO pins
    GPIO.setup(8, GPIO.OUT)   # CS
    GPIO.setup(25, GPIO.OUT)  # DC
    GPIO.setup(24, GPIO.OUT)  # RST
    GPIO.setup(11, GPIO.OUT)  # SCLK
    GPIO.setup(10, GPIO.OUT)  # MOSI
    
    # Create Pin objects for our GPIO pins
    cs_pin = digitalio.DigitalInOut(Pin(8))      # GPIO 8 (CE0)
    dc_pin = digitalio.DigitalInOut(Pin(25))     # GPIO 25
    reset_pin = digitalio.DigitalInOut(Pin(24))  # GPIO 24

    # Initialize SPI bus
    spi = busio.SPI(clock=Pin(11), MOSI=Pin(10))  # GPIO 11 (clock), GPIO 10 (MOSI)

    # Create the ILI9341 display:
    display = adafruit_ili9341.ILI9341(
        spi,
        rotation=90,  # Rotate display 90 degrees
        cs=cs_pin,
        dc=dc_pin,
        rst=reset_pin,
    )

    # Create blank image for drawing
    width = display.width
    height = display.height
    image = Image.new("RGB", (width, height))

    # Get drawing object to draw on image
    draw = ImageDraw.Draw(image)

    def test_display():
        # Fill the screen with red
        draw.rectangle((0, 0, width, height), fill=(255, 0, 0))
        display.image(image)
        time.sleep(1)
        
        # Fill the screen with green
        draw.rectangle((0, 0, width, height), fill=(0, 255, 0))
        display.image(image)
        time.sleep(1)
        
        # Fill the screen with blue
        draw.rectangle((0, 0, width, height), fill=(0, 0, 255))
        display.image(image)
        time.sleep(1)
        
        # Draw some shapes
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))  # Clear screen
        draw.rectangle((10, 10, 110, 110), fill=(255, 255, 255))  # White rectangle
        draw.ellipse((130, 10, 230, 110), fill=(255, 0, 0))  # Red circle
        draw.polygon([(10, 120), (110, 120), (60, 220)], fill=(0, 255, 0))  # Green triangle
        
        # Add some text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 230), "ILI9341 Test", font=font, fill=(255, 255, 255))
        display.image(image)

    if __name__ == "__main__":
        try:
            print("Testing display...")
            test_display()
            print("Test completed successfully!")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
