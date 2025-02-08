import board
import digitalio
import adafruit_rgb_display.ili9341 as ili9341
from PIL import Image, ImageDraw, ImageFont

class LCDDisplay:
    def __init__(self):
        self.display = self.init_display()
        self.load_fonts()

    def init_display(self):
        """Initialize the ILI9341 display"""
        # Configuration for CS and DC pins
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D24)
        
        # Configure SPI
        BAUDRATE = 24000000
        
        # Create the ILI9341 display
        spi = board.SPI()
        display = ili9341.ILI9341(
            spi,
            rotation=90,  # Adjust rotation as needed
            cs=cs_pin,
            dc=dc_pin,
            rst=reset_pin,
            baudrate=BAUDRATE,
        )
        
        # Enable backlight
        backlight = digitalio.DigitalInOut(board.D22)
        backlight.switch_to_output()
        backlight.value = True
        
        return display

    def load_fonts(self):
        """Load fonts for display"""
        try:
            self.main_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            self.main_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def update(self, incorrect_count=0, total_count=0):
        """Update the TFT display with warning and status information"""
        # Create a blank image with mode 'RGB'
        image = Image.new('RGB', (self.display.width, self.display.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw status information
        y_position = 10
        draw.text((10, y_position), "Plank Monitor", font=self.main_font, fill=(255, 255, 255))
        
        y_position += 40
        draw.text((10, y_position), f"Total: {total_count}", font=self.small_font, fill=(255, 255, 255))
        
        y_position += 30
        draw.text((10, y_position), f"Incorrect: {incorrect_count}", font=self.small_font, 
                fill=(255, 0, 0) if incorrect_count > 0 else (255, 255, 255))
        
        # Draw warning if there are incorrect planks
        if incorrect_count > 0:
            y_position += 50
            warning_text = "WARNING!"
            draw.text((10, y_position), warning_text, font=self.main_font, fill=(255, 0, 0))
            
            y_position += 40
            message = "Incorrect plank"
            message2 = "orientation detected!"
            draw.text((10, y_position), message, font=self.small_font, fill=(255, 255, 0))
            draw.text((10, y_position + 25), message2, font=self.small_font, fill=(255, 255, 0))
        
        # Display the image
        self.display.image(image)
