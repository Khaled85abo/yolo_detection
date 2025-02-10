import lgpio
import time

# Create an instance of the GPIO chip
h = lgpio.gpiochip_open(0)
print("GPIO chip initialized successfully!")

# Define GPIO pins
RED_LED = 17    # GPIO17
GREEN_LED = 18  # GPIO18

# Set up the GPIO pins as outputs
lgpio.gpio_claim_output(h, RED_LED)
lgpio.gpio_claim_output(h, GREEN_LED)
print("GPIO pins configured successfully!")
print("\nStarting LED alternation. Press Ctrl+C to stop.\n")

try:
    while True:
        # Turn on red LED, turn off green LED
        print("RED LED: ON  | GREEN LED: OFF")
        lgpio.gpio_write(h, RED_LED, 1)
        lgpio.gpio_write(h, GREEN_LED, 0)
        time.sleep(1)  # Wait for 1 second
        
        # Turn off red LED, turn on green LED
        print("RED LED: OFF | GREEN LED: ON")
        lgpio.gpio_write(h, RED_LED, 0)
        lgpio.gpio_write(h, GREEN_LED, 1)
        time.sleep(1)  # Wait for 1 second

except KeyboardInterrupt:
    print("\nProgram stopped by user")
    
finally:
    # Clean up GPIO on program exit
    print("Cleaning up GPIO...")
    lgpio.gpio_write(h, RED_LED, 0)
    lgpio.gpio_write(h, GREEN_LED, 0)
    lgpio.gpiochip_close(h)
    print("Cleanup complete. Program ended.")
