import RPi.GPIO as GPIO
import spidev
import time

# Pin definitions
DC = 25    # Data/Command pin
RST = 24   # Reset pin
CS = 8     # Chip Select (using CE0)

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(DC, GPIO.OUT)
GPIO.setup(RST, GPIO.OUT)

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, CE0
spi.max_speed_hz = 32000000  # Start with 32MHz
spi.mode = 0

def command(cmd):
    GPIO.output(DC, 0)  # Command mode
    spi.xfer([cmd])

def data(data):
    GPIO.output(DC, 1)  # Data mode
    spi.xfer([data])

def init_display():
    # Reset display
    GPIO.output(RST, 0)
    time.sleep(0.1)
    GPIO.output(RST, 1)
    time.sleep(0.1)

    # Initialize display
    command(0x01)  # Software reset
    time.sleep(0.1)
    
    command(0xCF)  # Power control B
    data(0x00)
    data(0xC1)
    data(0x30)
    
    command(0xED)  # Power on sequence control
    data(0x64)
    data(0x03)
    data(0x12)
    data(0x81)
    
    command(0xE8)  # Driver timing control A
    data(0x85)
    data(0x00)
    data(0x78)
    
    command(0xCB)  # Power control A
    data(0x39)
    data(0x2C)
    data(0x00)
    data(0x34)
    data(0x02)
    
    command(0xF7)  # Pump ratio control
    data(0x20)
    
    command(0xEA)  # Driver timing control B
    data(0x00)
    data(0x00)
    
    command(0xC0)  # Power Control 1
    data(0x23)
    
    command(0xC1)  # Power Control 2
    data(0x10)
    
    command(0xC5)  # VCOM Control 1
    data(0x3E)
    data(0x28)
    
    command(0xC7)  # VCOM Control 2
    data(0x86)
    
    command(0x36)  # Memory Access Control
    data(0x48)     # RGB color filter panel
    
    command(0x3A)  # Pixel Format Set
    data(0x55)     # 16-bit color
    
    command(0xB1)  # Frame Rate Control
    data(0x00)
    data(0x18)
    
    command(0xB6)  # Display Function Control
    data(0x08)
    data(0x82)
    data(0x27)
    
    command(0xF2)  # 3Gamma Function Disable
    data(0x00)
    
    command(0x26)  # Gamma curve selected
    data(0x01)
    
    command(0x11)  # Exit Sleep
    time.sleep(0.120)
    
    command(0x29)  # Display on
    time.sleep(0.120)

try:
    print("Initializing display...")
    init_display()
    
    # Fill screen with red color
    command(0x2C)  # Memory Write
    for y in range(320):
        for x in range(240):
            data(0xF8)  # Red (high byte)
            data(0x00)  # Red (low byte)
            
    print("Display test complete - screen should be red")
    
except Exception as e:
    print(f"Error: {e}")
    
finally:
    GPIO.cleanup()
    spi.close()
