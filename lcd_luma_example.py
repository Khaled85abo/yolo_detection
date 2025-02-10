from luma.core.interface.serial import spi
from luma.lcd.device import ili9341
from PIL import Image, ImageDraw, ImageFont
import RPi.GPIO as GPIO

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Initialize SPI with GPIO interface
serial = spi(port=0, device=0, gpio=GPIO, gpio_DC=25, gpio_RST=24)
device = ili9341(serial, width=240, height=320)

image = Image.new("RGB", (240, 320))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((10, 10), "Hello luma.lcd!", font=font, fill="white")

device.display(image)
