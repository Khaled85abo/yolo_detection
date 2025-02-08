import board
import busio
import digitalio
from adafruit_ili9341 import ILI9341
from PIL import Image, ImageDraw, ImageFont

# Initialize SPI
spi = busio.SPI(board.SCK, board.MOSI)

# Define control pins (adjust for your wiring)
cs_pin = digitalio.DigitalInOut(board.CE0)   # or board.CE1
dc_pin = digitalio.DigitalInOut(board.D25)   # pick a free GPIO
reset_pin = digitalio.DigitalInOut(board.D24)  # pick a free GPIO

# Create display object
display = ILI9341(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=32000000,   # try a high speed like 31-32 MHz
    width=240,
    height=320,
    rotation=0           # set rotation as needed
)

# Clear screen to black
display.fill(0)

# Create a PIL image to draw on
image = Image.new("RGB", (240, 320))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Draw white text
draw.text((10, 10), "Hello ILI9341!", font=font, fill=(255, 255, 255))

# Push the image to the display
display.image(image)
