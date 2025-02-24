#include <TFT_eSPI.h>
#include <lvgl.h>
// For a hypothetical XPT2046 library
#include <XPT2046_Touchscreen.h>

TFT_eSPI tft = TFT_eSPI(); // Create TFT instance

// LVGL display flushing function
void my_disp_flush(lv_disp_drv_t *disp, const lv_area_t *area, lv_color_t *color_p)
{
  // 'area' describes the region to draw
  // 'color_p' is a pointer to the pixel color data

  // Set address window in TFT_eSPI for the region
  tft.startWrite();
  tft.setAddrWindow(area->x1, area->y1, (area->x2 - area->x1 + 1), (area->y2 - area->y1 + 1));
  // Write the color data
  tft.pushColors((uint16_t *)&color_p->full, (area->x2 - area->x1 + 1) * (area->y2 - area->y1 + 1), true);
  tft.endWrite();

  // Indicate you are done
  lv_disp_flush_ready(disp);
}

void setup()
{
  Serial.begin(115200);

  // Initialize TFT
  tft.init();
  tft.setRotation(1); // or the orientation you want

  // Initialize LVGL
  lv_init();

  // Create a buffer for 1/10 screen size or so
  static lv_color_t buf1[LV_HOR_RES_MAX * 10];
  static lv_disp_draw_buf_t draw_buf;
  lv_disp_draw_buf_init(&draw_buf, buf1, NULL, LV_HOR_RES_MAX * 10);

  // Register display
  static lv_disp_drv_t disp_drv;
  lv_disp_drv_init(&disp_drv);
  disp_drv.hor_res = 240; // your TFT width
  disp_drv.ver_res = 320; // your TFT height
  disp_drv.flush_cb = my_disp_flush;
  disp_drv.draw_buf = &draw_buf;
  lv_disp_drv_register(&disp_drv);

  // If you have a touch driver, register an input device similarly...

  // Now create a simple LVGL object
  lv_obj_t *label = lv_label_create(lv_scr_act());
  lv_label_set_text(label, "Hello, LVGL!");
  lv_obj_align(label, LV_ALIGN_CENTER, 0, 0);
}

void loop()
{
  // lv_timer_handler periodically
  lv_timer_handler(); // must be called periodically (every few ms)
  delay(5);
}

// Create a button
lv_obj_t *btn = lv_btn_create(lv_scr_act());
lv_obj_align(btn, LV_ALIGN_CENTER, 0, 0);

// Attach a label
lv_obj_t *label = lv_label_create(btn);
lv_label_set_text(label, "Press Me!");

// Add an event callback
lv_obj_add_event_cb(btn, btn_event_cb, LV_EVENT_ALL, NULL);

static void btn_event_cb(lv_event_t *e)
{
  // Check event type
  lv_event_code_t code = lv_event_get_code(e);
  if (code == LV_EVENT_CLICKED)
  {
    Serial.println("Button clicked!");
  }
}

// // For a hypothetical XPT2046 library
// #include <XPT2046_Touchscreen.h>

// Create the touch instance
XPT2046_Touchscreen touch(/*csPin=*/5 /* or your pin */);

static void my_input_read(lv_indev_drv_t *drv, lv_indev_data_t *data)
{
  if (touch.touched())
  {
    TS_Point p = touch.getPoint();
    data->point.x = p.x;
    data->point.y = p.y;
    data->state = LV_INDEV_STATE_PRESSED;
  }
  else
  {
    data->state = LV_INDEV_STATE_RELEASED;
  }
}

void setup()
{
  // ... other init ...
  touch.begin();

  static lv_indev_drv_t indev_drv;
  lv_indev_drv_init(&indev_drv);
  indev_drv.type = LV_INDEV_TYPE_POINTER;
  indev_drv.read_cb = my_input_read;
  lv_indev_drv_register(&indev_drv);
  // Now LVGL knows how to get touch info
}
